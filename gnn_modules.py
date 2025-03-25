'''
© 2025 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, TransformerConv, Linear, LayerNorm
from gnn_utils import _compute_batch_EE
from tqdm import tqdm


# LightningDataModule containing the graph data
class GraphDataModule(pl.LightningDataModule):
    # files_info contains a list of tuples. Each tuple is of the form
    # (filename, is_train, is_val, val_batch_size), where:
    #      - filename is the name of file containing the preprocessed graph
    #        data
    #      - is_train indicates whether this dataset is used for training
    #      - is_val indicates whether this dataset is used for validation
    #      - val_batch_size is the validation batch size used when is_val==True
    def __init__(self, files_info, tr_batch_size, num_workers, precision=None,
                 verbose=False):
        super().__init__()
        self.files_info = files_info
        self.tr_batch_size = tr_batch_size
        self.num_workers = num_workers
        self.precision = precision

        # Generate names for the train and val datasets
        self.tr_names = []
        self.val_names = []
        if verbose:
            print('Start loading files in GraphDataModule')
        for i in tqdm(range(len(files_info)), disable=not verbose):
            filename, is_train, is_val, _ = files_info[i]
            file = torch.load(filename)
            n_ues = file['n_ues']
            n_aps = file['n_aps']

            name = "{}aps_{}ues".format(n_aps, n_ues)

            if is_train:
                self.tr_names.append(name)
            if is_val:
                self.val_names.append(name)

    def setup(self, stage=None):
        # Load the preprocessed and splitted datasets
        self.train_dataset = []
        self.val_datasets = []
        self.test_datasets = []

        for i in range(len(self.files_info)):
            filename, is_train, is_val, val_batch_size = self.files_info[i]
            file = torch.load(filename)

            # Prepare the training and validation datasets
            if stage is None or stage == "fit":
                if is_train:
                    self.train_dataset.extend(file['train_data'])
                if is_val:
                    self.val_datasets.append((file['val_data'],
                                              val_batch_size))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.tr_batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return [DataLoader(tup[0], batch_size=tup[1], shuffle=False,
                num_workers=self.num_workers) for tup in self.val_datasets]


# This LightningModule defines the base model's training_step, validation_step,
# and logging functions. The layers and forward function are defined in the
# child classes.
# input:
#   - model_name:   name of the nn model
#   - tr_names:     list of names for the training datasets
#   - val_names:    list of names for the validation datasets
#   - lr:           learning rate
#   - optim:        optimizer
#   - batch_size:   batch size
#   - loss_alpha:   coefficient used in the mse loss function, such that:
#                   loss = sinr_loss + loss_alpha*y_loss
class BaseGNNModule(pl.LightningModule):

    def __init__(self, model_name, tr_names, val_names, lr, optim, batch_size,
                 loss_alpha=0.0, **kwargs):
        super().__init__()

        # Save the arguments to hparams to be logged as hyperparemeters
        self.save_hyperparameters("model_name", "tr_names", "val_names", "lr",
                                  "optim", "batch_size", "loss_alpha")
        self.save_hyperparameters(kwargs)

        # Parameters
        self.lr = lr
        self.optim = optim
        self.val_names = val_names
        self.loss_alpha = loss_alpha

        self.val_step_outputs = [[] for x in range(len(val_names))]

    def common_step(self, batch):
        y_hat = self(batch)
        y = batch['channel'].y
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.common_step(batch)
        y_loss = F.mse_loss(y_hat, y, reduction='mean')

        EE, EE_hat, SE, SE_hat = _compute_batch_EE(batch, y_hat)

        EE_loss = F.mse_loss(EE, EE_hat, reduction='mean')
        EE_rel_loss = (EE-EE_hat)/EE

        QoS_Constraints = torch.ones(SE_hat.shape, device=SE_hat.device)
        QoS_loss = torch.mean(F.relu(QoS_Constraints - SE_hat))

        # SE_loss without ReLU converges faster and reaches better EE
        # SE_loss = torch.mean(F.relu(SE_hat - SE))
        SE_loss = F.mse_loss(SE, SE_hat, reduction='mean')
        SE_rel_loss = torch.mean(torch.abs(SE-SE_hat)/SE)

        loss = y_loss + 0.25 * SE_loss 

        self.log("EE_loss", EE_loss, prog_bar=True)

        self.log("EE_rel_loss", EE_rel_loss, prog_bar=True)
        self.log("SE_loss", SE_loss, prog_bar=True)
        self.log("SE_rel_loss", SE_rel_loss, prog_bar=True)
        self.log("QoS_loss", QoS_loss, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch.num_graphs

        # Log mse loss of the validation data
        y_hat, y = self.common_step(batch)

        y_loss = F.mse_loss(y_hat, y, reduction='mean')
        val_loss = y_loss
        self.log("val_loss/{}".format(self.val_names[dataloader_idx]),
                 val_loss, add_dataloader_idx=False, batch_size=batch_size)

        # Log the SINR performance loss (relative and mse)
        EE, EE_hat, SE, SE_hat = _compute_batch_EE(batch, y_hat)
        EE_loss = F.mse_loss(EE, EE_hat, reduction='mean')
        EE_rel_loss = (EE-EE_hat)/EE
        
        SE_loss = F.mse_loss(SE, SE_hat, reduction='mean')
        SE_rel_loss = torch.mean(torch.abs(SE-SE_hat)/SE)
        
        QoS_Constraints = torch.ones(SE_hat.shape, device=SE_hat.device)
        QoS_loss = torch.mean(F.relu(QoS_Constraints - SE_hat))

        # Save the validation loss on this dataset to be used in the method
        # on_validation_epoch_end()

        self.val_step_outputs[dataloader_idx].append(val_loss)

        self.log("val_EE_loss/{}".format(
            self.val_names[dataloader_idx]), EE_loss,
            add_dataloader_idx=False, batch_size=batch_size)

        self.log("val_rel_EE_loss/{}".format(
            self.val_names[dataloader_idx]), EE_rel_loss,
            add_dataloader_idx=False, batch_size=batch_size)

        self.log("val_SE_loss/{}".format(
            self.val_names[dataloader_idx]), SE_loss,
            add_dataloader_idx=False, batch_size=batch_size)

        self.log("val_rel_SE_loss/{}".format(
            self.val_names[dataloader_idx]), SE_rel_loss,
            add_dataloader_idx=False, batch_size=batch_size)
        
        self.log("val_QoS_loss/{}".format(
            self.val_names[dataloader_idx]), QoS_loss,
            add_dataloader_idx=False, batch_size=batch_size)

        loss = y_loss + 0.25*SE_loss
        return loss

    def predict_step(self, batch, batch_idx, dataset_idx=0):
        return batch, self(batch)

    # Log the average loss over all validation datasets (outputs of all
    # validation_step calls)
    def on_validation_epoch_end(self):
        flat_list = []
        for idx in range(len(self.val_step_outputs)):
            flat_list.extend(self.val_step_outputs[idx])
            # Free memory
            self.val_step_outputs[idx].clear()
        avg_loss = sum(flat_list) / len(flat_list)
        self.log("hp_metric", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        if self.optim == 'NAdam':
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        elif self.optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            assert False, ("Argument optim={} is not supported. Valid values "
                           "are \'NAdam\' and \'Adam\'").format(self.optim)
        return optimizer


class EEBaseGNNModule(BaseGNNModule):
    '''
    Base/asbtract class of the GNN for Energy Efficiency (EE) maximization.
    self.edge_attr_update must be implemented by any child class.
    '''

    def __init__(self, model_name_suffix, tr_names, val_names, lr, optim,
                 num_epochs, batch_size, hc, heads, loss_alpha=0.0, aggr='max',
                 float_precision=32):
        model_name = "EE-GNN" + model_name_suffix
        super().__init__(model_name, tr_names, val_names, lr, optim,
                         batch_size, loss_alpha=loss_alpha,
                         num_epochs=num_epochs,
                         float_precision=float_precision,
                         heads=heads, hc=hc, aggr=aggr)
        num_layers = len(hc)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(num_layers-1):
            in_channels = hc[i]
            out_channels = int(hc[i+1]/heads)
            conv = HeteroConv({
                ('ap_node', 'edge', 'ue_node'):
                    TransformerConv(in_channels, out_channels, heads=heads,
                                    edge_dim=in_channels),
                ('ue_node', 'edge', 'ap_node'):
                    TransformerConv(in_channels, out_channels, heads=heads,
                                    edge_dim=in_channels)},
                    aggr=aggr)
            self.convs.append(conv)

            self.norms.append(LayerNorm(hc[i+1]))

        self.lin = Linear(hc[-1], 1)

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        if hasattr(batch['ue_node'], 'batch'):
            ue_node_batch = batch['ue_node'].batch
        else:
            ue_node_batch = None

        if hasattr(batch['ap_node'], 'batch'):
            ap_node_batch = batch['ap_node'].batch
        else:
            ap_node_batch = None

        for it, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

            edge_attr_dict = self.edge_attr_update(x_dict, edge_attr_dict,
                                                   edge_index_dict, it)

            x_dict = {'ap_node': norm(x_dict['ap_node'].relu(), ap_node_batch),
                      'ue_node': norm(x_dict['ue_node'].relu(), ue_node_batch)}

        return self.lin(edge_attr_dict['ap_node', 'edge', 'ue_node'])


class EEGNNModule(EEBaseGNNModule):
    '''
    Concrete class of the GNN for Energy Efficiency (EE) maximization.
    '''

    def __init__(self, tr_names, val_names, lr, optim, num_epochs, batch_size,
                 hc, heads, loss_alpha=0.0, aggr='max',
                 float_precision=32):
        model_name_suffix = "_additive_edge_updates"
        super().__init__(model_name_suffix, tr_names, val_names, lr, optim,
                         num_epochs, batch_size, hc, heads, loss_alpha, aggr,
                         float_precision)

        num_layers = len(hc)
        self.lin_ap_ue_src = torch.nn.ModuleList()
        self.lin_ap_ue_dst = torch.nn.ModuleList()
        self.lin_ap_ue_self = torch.nn.ModuleList()
        self.lin_ue_ap_src = torch.nn.ModuleList()
        self.lin_ue_ap_dst = torch.nn.ModuleList()
        self.lin_ue_ap_self = torch.nn.ModuleList()
        for i in range(num_layers-1):
            in_channels = hc[i]
            out_channels = hc[i+1]
            self.lin_ap_ue_src.append(Linear(out_channels, out_channels))
            self.lin_ap_ue_dst.append(Linear(out_channels, out_channels))
            self.lin_ap_ue_self.append(Linear(in_channels, out_channels))
            self.lin_ue_ap_src.append(Linear(out_channels, out_channels))
            self.lin_ue_ap_dst.append(Linear(out_channels, out_channels))
            self.lin_ue_ap_self.append(Linear(in_channels, out_channels))

    def edge_attr_update(self, x_dict, edge_attr_dict, edge_index_dict, it):
        new_edge_attr_dict = {}

        # Edge of type ['ap_node', 'edge', 'ue_node']
        source_nodes = x_dict['ap_node'][
            edge_index_dict['ap_node', 'edge', 'ue_node'][0, :]]
        destination_nodes = x_dict['ue_node'][
            edge_index_dict['ap_node', 'edge', 'ue_node'][1, :]]
        new_edge_attr_dict['ap_node', 'edge', 'ue_node'] = \
            self.lin_ap_ue_src[it](source_nodes) + \
            self.lin_ap_ue_dst[it](destination_nodes) + \
            self.lin_ap_ue_self[it](
                edge_attr_dict['ap_node', 'edge', 'ue_node'])

        # Edge of type ['ue_node', 'edge', 'ap_node']
        source_nodes = x_dict['ue_node'][
            edge_index_dict['ue_node', 'edge', 'ap_node'][0, :]]
        destination_nodes = x_dict['ap_node'][
            edge_index_dict['ue_node', 'edge', 'ap_node'][1, :]]
        new_edge_attr_dict['ue_node', 'edge', 'ap_node'] = \
            self.lin_ue_ap_src[it](source_nodes) + \
            self.lin_ue_ap_dst[it](destination_nodes) + \
            self.lin_ue_ap_self[it](
                edge_attr_dict['ue_node', 'edge', 'ap_node'])

        # Activation function
        for edge_type, edge_attr in new_edge_attr_dict.items():
            new_edge_attr_dict[edge_type] = edge_attr.relu()

        # Normalization layer can be added here is needed

        return new_edge_attr_dict

'''
© 2025 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import torch
import pytorch_lightning as pl
from gnn_modules import GraphDataModule
from gnn_modules import EEGNNModule


torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

# Model training parameters
NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 7e-4
OPTIM = 'Adam'  # 'NAdam' or 'Adam'

# Other parameters
FLOAT_PRECISION = 32
DATALOADER_NUM_WORKERS = 4

MODEL_NAME = "EEGNNModule"

# Fixed seed
pl.seed_everything(0)

# Load data
val_batch_size = 16

files_info = [('preprocessed_graph_data/EE_50aps_10ues.pt',
               True, True, val_batch_size),
              ('preprocessed_graph_data/EE_60aps_15ues.pt',
               True, True, val_batch_size),
              ('preprocessed_graph_data/EE_75aps_25ues.pt',
               True, True, val_batch_size),
              ('preprocessed_graph_data/EE_100aps_30ues.pt',
               True, True, val_batch_size),
              ('preprocessed_graph_data/EE_200aps_10ues.pt',
               False, True, val_batch_size),
              ('preprocessed_graph_data/EE_200aps_20ues.pt',
               False, True, val_batch_size),
              ('preprocessed_graph_data/EE_200aps_30ues.pt',
               False, True, val_batch_size),
              ('preprocessed_graph_data/EE_200aps_40ues.pt',
               False, True, val_batch_size),
            ]

datamodule = GraphDataModule(files_info,
                             tr_batch_size=TRAIN_BATCH_SIZE,
                             num_workers=DATALOADER_NUM_WORKERS,
                             precision=FLOAT_PRECISION,
                             verbose=True)

# Get model
if MODEL_NAME == "EEGNNModule":
    aggr = "sum"
    heads = 1

    hc = [1, 8, 16, 16, 32, 64, 64, 32, 16, 16, 8, 1]

    print('aggr={}'.format(aggr))
    print('heads={}'.format(heads))
    print('hc={}'.format(hc))
    model = EEGNNModule(
        datamodule.tr_names, datamodule.val_names, LEARNING_RATE, hc=hc,
        heads=heads, optim=OPTIM, num_epochs=NUM_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE, float_precision=FLOAT_PRECISION,
        aggr=aggr)
else:
    assert False, "Unknown model name {}".format(MODEL_NAME)

# Fit
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="hp_metric",
                                                   save_last=True,
                                                   every_n_epochs=1,
                                                   save_top_k=-1)


trainer = pl.Trainer(devices=[0], accelerator="gpu",
                     max_epochs=NUM_EPOCHS, precision=FLOAT_PRECISION,
                     callbacks=[checkpoint_callback], num_sanity_val_steps=0)

trainer.fit(model, datamodule)

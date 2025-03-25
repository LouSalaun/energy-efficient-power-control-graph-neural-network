'''
© 2025 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import os
import sys
import time
import torch
import numpy as np
import torch_geometric
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from gnn_modules import EEGNNModule
from gnn_utils import _compute_batch_EE
import matplotlib.pyplot as plt
import seaborn as sns


# Ignore warnings if needed
import warnings
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')

# Create folder to save figures and csv results
os.makedirs('test_results', exist_ok=True)

# Parameters
FLOAT_PRECISION = 32
DATALOADER_NUM_WORKERS = 0

# Load the model from the provided path
PATH = sys.argv[1]
model_name = "EEGNNModule"
model = EEGNNModule.load_from_checkpoint(PATH)

test_device = 'cuda'  # or 'cpu'
device = torch.device(test_device)
model = model.to(device)
     
trainer = pl.Trainer(accelerator=test_device,precision=FLOAT_PRECISION,
                     logger=False, enable_checkpointing=False)               

preprocessed_folder = 'preprocessed_graph_data'
filenames = [os.path.join(preprocessed_folder, fn)
             for fn in os.listdir(preprocessed_folder)
             if fn.endswith('.pt')]

batch_sizes = 1*np.ones(int(len(filenames)), dtype=int)
batch_sizes = batch_sizes.tolist()
styles = [{"linestyle": '-'},{"linestyle": '-.'}]

results_csv_rows = [['Number of APs (M)', 'Number of UEs (K)',
                     'Median EE loss (%)']]
for i, filename in enumerate(filenames):
    print("----------------------------------------")
    print("Testing data from {}".format(filename))
    # Load preprocessed data
    file = torch.load(filename)
    data_list = file['test_data']
    n_data = len(data_list)
    n_ues = file['n_ues']
    n_aps = file['n_aps']
    input_mean = file['input_mean']
    input_std = file['input_std']
    output_mean = file['output_mean']
    output_std = file['output_std']
    gamma_mean = file['gamma_mean']
    gamma_std = file['output_std']

    batch_size = batch_sizes[i]
    dataloader = DataLoader(data_list, batch_size=batch_size,
                            shuffle=False, num_workers=DATALOADER_NUM_WORKERS)
                
    start_time = time.process_time_ns()
    pred = trainer.predict(model, dataloader)

    end_time = time.process_time_ns()
    prediction_time = (end_time-start_time)/(10 ** 9)
    print("Prediction Time:", prediction_time)

    n_features = n_ues*n_aps
    avg_ee_loss = 0
    avg_se_viol_hat = 0
    avg_se_viol = 0

    ee = torch.zeros(n_data)
    ee_hat = torch.zeros(n_data)
    for batch_idx, tup in enumerate(pred):
        batch, batch_y_hat = tup
        b_ee, b_ee_hat, b_sinr, b_sinr_hat = \
            _compute_batch_EE(batch, batch_y_hat)
        ee[batch_idx] = b_ee
        ee_hat[batch_idx] = b_ee_hat
        avg_ee_loss += (b_ee-b_ee_hat)/b_ee
        avg_se_viol_hat += torch.sum(b_sinr_hat<1)
        avg_se_viol += torch.sum(b_sinr<1)

    m_ee = torch.median(ee)
    m_ee_hat = torch.median(ee_hat)
    ee_rel_loss = (m_ee - m_ee_hat) / m_ee
    results_csv_rows.append([n_aps, n_ues,
                             '{:.2%}'.format(ee_rel_loss)])
    print('median EE perf loss = {:.2%}'.format(ee_rel_loss))
    print('avg GNN QoS violation = {}'.format(avg_se_viol_hat/len(pred)))
    print('avg APG QoS violation = {}'.format(avg_se_viol/len(pred)))

    plt.figure()
    sns.ecdfplot(ee, **styles[0], label='True')
    sns.ecdfplot(ee_hat, **styles[1], label='Predicted')
    plt.ylabel('CDF')
    plt.xlabel('Energy Efficiency')
    plt.savefig('test_results/ee_{}aps_{}ues.png'.format(n_aps, n_ues))

import csv
with open('test_results/ee_results.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerows(results_csv_rows)

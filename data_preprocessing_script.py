'''
© 2025 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

from data_preprocessing import preprocess_data_EE
import os
import sys
import json


def preprocess_with_info_dict(info_file):
    # Open the json file
    info_dict = json.load(open(info_file))
    data_path = os.path.dirname(info_file)

    # Normalization based on the urban 32x9 use-case statistics
    input_mean = -41.40197
    input_std = 3.66350
    output_mean = -6.59817
    output_std = 3.03213
    gamma_mean = -43.17697
    gamma_std = 5.34856

    n_aps = info_dict['n_aps']
    n_ues = info_dict['n_ues']
    save_filename = \
        'preprocessed_graph_data/EE_{}aps_{}ues.pt'.format(n_aps, n_ues)
    betas = os.path.join(data_path, info_dict['beta'])
    gammas = os.path.join(data_path, info_dict['gamma'])
    thetas = os.path.join(data_path, info_dict['theta'])
    pilots = os.path.join(data_path, info_dict['pilots'])
    n_val = info_dict['n_val']
    n_test = info_dict['n_test']

    print('------------------------------')
    print('Preprocessing scenario EE_{}aps_{}ues'.format(n_aps, n_ues))
    print('------------------------------')

    preprocess_data_EE(betas, gammas, thetas, pilots, save_filename, n_ues,
                       n_aps, n_val=n_val, n_test=n_test, l_lim=1e-70,
                       input_mean=input_mean, input_std=input_std,
                       output_mean=output_mean, output_std=output_std,
                       gamma_mean=gamma_mean, gamma_std=gamma_std,
                       verbose=True)


# Create 'preprocessed_graph_data' folder if it does not already exists
os.makedirs('preprocessed_graph_data', exist_ok=True)

# Path to the folder containing all datasets
data_path = sys.argv[1]

# Search 'info.json' in data_path and its subfolders
for tup in os.walk(data_path):
    info_file = os.path.join(tup[0], "info.json")
    if os.path.isfile(info_file):
        preprocess_with_info_dict(info_file)

'''
© 2025 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm


def _parse_file(file, n_rows, n_cols, delimiter=' '):
    """
    Load files, reshape and store them

    Args:
        file (txt): filename to be loaded
        n_rows (int): number of rows = n_APs
        n_cols (int): number of columns = n_UEs

    Returns:
        res(tensor): reshaped files converted to tensors
    """
    ar = torch.from_numpy(np.loadtxt(file, delimiter=delimiter))
    n_vals = ar.shape[0] * ar.shape[1]
    n_samples, r = divmod(n_vals, n_rows*n_cols)
    assert r == 0, ("The number of values in file {} is inconsistent with "
                    "n_rows = {} and n_cols = {}").format(file, n_rows, n_cols)
    res = torch.zeros((n_samples, n_rows, n_cols))
    for i in range(n_samples):
        res[i] = ar[n_rows*i:n_rows*(i+1)]
    return res


def _normalize_arrays(ar, log2=True, normalization=True, mean=None, std=None,
                      l_lim=None):
    """
    Normalize arrays with mean and std of the 32x9 case

    Args:
        ar (tensor): input array to be normalized
        log2 (bool, optional): log transform of raw data. Defaults to True.
        normalization (bool, optional): normalize with given input mean and std
                                        Defaults to True.
        mean (float, optional): mean of the 32x9 case. Defaults to None.
        std (float, optional): std of the 32x9 case. Defaults to None.
        l_lim (double, optional): lower limit for clipping the data.
                                  Defaults to None.

    Returns:
        preprocessed_arrays (torch): array after preprocessing
        mean(float): calculated mean, if different from input mean
        std(float): calculated std, if different from input std
    """
    ar = torch.clip(ar, l_lim, None)

    # Log2 transformation
    if log2:
        preprocessed_arrays = torch.log2(ar)
    else:
        preprocessed_arrays = ar

    # Normalization
    if normalization:
        if mean is None:
            mean = preprocessed_arrays.mean()
        if std is None:
            std = preprocessed_arrays.std()
        preprocessed_arrays -= mean
        preprocessed_arrays /= std

    return preprocessed_arrays, mean, std


def preprocess_data_EE(beta_file, gamma_file, theta_file, pilots_file,
                       save_filename, n_ues, n_aps, n_val, n_test, l_lim,
                       input_mean=None, input_std=None,
                       output_mean=None, output_std=None,
                       gamma_mean=None, gamma_std=None, verbose=True):
    """
    Convert np data to Heterodata graph representation and saves to
    save_filename.

    Args:
        files (type): source files.
        save_filename (type): destination .pt graph file.
        n_ues (int): number of UEs.
        n_aps (int): number of APs.
        n_val (int): number of validation samples.
        n_test (int): number of test samples.
        l_lim (double): lower limit for clipping the data.
        input_mean (float, optional): mean value used to normalize the input
                                      beta. Defaults to None.
        input_std (float, optional): std value used to normalize the input
                                     beta. Defaults to None.
        output_mean (float_, optional): mean value used to normalize the output
                                        theta. Defaults to None.
        output_std (float, optional): std value used to normalize the output
                                      theta. Defaults to None.
        gamma_mean (float, optional): mean value used to normalize the input
                                      gamma. Defaults to None.
        gamma_std (float, optional): std value used to normalize the input
                                     gamma. Defaults to None.
        verbose (bool, optional): Defaults to True.
    """

    # ---------- Convert to graph data ----------
    if verbose:
        print('------------------------------')
        print('Start converting the data into graph format!')

    # Convert each sample to a HeteroData object
    graphs_data = []

    beta_preproc = _parse_file(beta_file, n_aps, n_ues)
    gamma_preproc = _parse_file(gamma_file, n_aps, n_ues)
    theta_preproc = _parse_file(theta_file, n_aps, n_ues)
    pilots_list = _parse_file(pilots_file, n_ues, n_ues)

    # ---------- Preprocessing ----------
    if verbose:
        print('------------------------------')
        print('Start normalization!')
    beta_list, input_mean, input_std =\
        _normalize_arrays(beta_preproc, l_lim=l_lim,
                          mean=input_mean, std=input_std)
    theta_list, output_mean, output_std =\
        _normalize_arrays(theta_preproc, l_lim=l_lim,
                          mean=output_mean, std=output_std)
    gamma_list, gamma_mean, gamma_std =\
        _normalize_arrays(gamma_preproc, l_lim=l_lim,
                          mean=gamma_mean, std=gamma_std)
    if verbose:
        print('Normalize input data with mean={}, std={}'
              .format(input_mean, input_std))
        print('Normalize output data with mean={}, std={}'
              .format(output_mean, output_std))
        print('Normalize gamma data with mean={}, std={}'
              .format(gamma_mean, gamma_std))
        print('Normalization done!')

    n_samples = len(beta_list)

    ue_ap_edges = []
    for i in range(n_ues):
        for j in range(n_aps):
            ue_ap_edges.append([i, j])

    ue_ap_edges = torch.tensor(ue_ap_edges, dtype=torch.long)
    ue_ap_edges = ue_ap_edges.t().contiguous()

    ap_ue_edges = []
    for i in range(n_aps):
        for j in range(n_ues):
            ap_ue_edges.append([i, j])

    ap_ue_edges = torch.tensor(ap_ue_edges, dtype=torch.long)
    ap_ue_edges = ap_ue_edges.t().contiguous()

    for i in tqdm(range(n_samples), disable=not verbose):

        beta = beta_list[i]
        beta_ap_ue = torch.reshape(beta, [-1, 1])
        beta_ue_ap = torch.reshape(beta.T, [-1, 1])
        beta = beta_ap_ue

        gamma = gamma_list[i]
        gamma = torch.reshape(gamma, [-1, 1])

        theta = theta_list[i]
        theta = torch.reshape(theta, [-1, 1])

        pilots = torch.tensor(pilots_list[i], dtype=torch.float32)
        pilots = torch.reshape(pilots, [-1, 1])

        data = HeteroData()
        data['ue_node'].x = torch.zeros((n_ues, 1))
        data['ap_node'].x = torch.zeros((n_aps, 1))

        data['channel'].gamma = gamma
        data['channel'].beta = beta
        data['channel'].y = theta
        data['channel'].pilots = pilots

        data['ue_node', 'edge', 'ap_node'].edge_index = ue_ap_edges
        data['ap_node', 'edge', 'ue_node'].edge_index = ap_ue_edges

        data['ue_node', 'edge', 'ap_node'].edge_attr = beta_ue_ap
        data['ap_node', 'edge', 'ue_node'].edge_attr = beta_ap_ue

        # Add metadata
        data['channel'].n_ues = n_ues
        data['channel'].n_aps = n_aps
        data['channel'].input_mean = input_mean
        data['channel'].input_std = input_std
        data['channel'].output_mean = output_mean
        data['channel'].output_std = output_std
        data['channel'].gamma_mean = gamma_mean
        data['channel'].gamma_std = gamma_std

        graphs_data.append(data)

    if verbose:
        print('Done!')

    # ---------- Divide into train, validation and test datasets ----------
    n_train = n_samples - n_val - n_test
    assert n_train >= 0, ("Invalid n_val and n_test for datasets "
                          "with only {} raw samples").format(n_samples)
    train_data = graphs_data[:n_train]
    val_data = graphs_data[n_train:n_train+n_val]
    test_data = graphs_data[n_train+n_val:]

    # ---------- Store the preprocessed graph data ----------
    res = {'train_data': train_data, 'val_data': val_data,
           'test_data': test_data, 'n_ues': n_ues, 'n_aps': n_aps,
           'input_mean': input_mean, 'input_std': input_std,
           'output_mean': output_mean, 'output_std': output_std,
           'gamma_mean': gamma_mean, 'gamma_std': gamma_std}

    torch.save(res, save_filename)
    if verbose:
        print('------------------------------')
        print('Preprocessing completed! Data saved in', save_filename)
        print('------------------------------')

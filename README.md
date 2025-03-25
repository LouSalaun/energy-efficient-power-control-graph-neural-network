# Energy Efficient Power Control Graph Neural Network

This repository contains the code of our paper "Bigraph GNN-Aided Energy Efficiency Maximization for Cell-Free Massive MIMO", in *IEEE International Conference on Machine Learning for Communication and Networking* (ICMLCN), 2025. The datasets are available at this address: https://doi.org/10.6084/m9.figshare.28639649.

© 2025 Nokia\
Licensed under the BSD 3-Clause Clear License\
SPDX-License-Identifier: BSD-3-Clause-Clear

## Dependencies

This program is implemented in Python 3.12 with PyTorch 2.5, Pytorch Geometric 2.6 and
PyTorch Lightning 2.5. The full list of packages can be found in `requirements.txt`. Run `pip install -r requirements.txt` to install them.

## Quickstart

Here are quick instructions to reproduce the numerical results presented in our paper.
1. First download the raw datasets from https://doi.org/10.6084/m9.figshare.28639649 and place them in a separate folder, e.g., named `data`.
2. Run the following script to preprocess and split the data into training, validation and test datasets, which are saved in a folder `preprocessed_graph_data` for later use:

    ```python data_preprocessing_script.py data```

3. A trained model is provided in `trained_model/checkpoints/best_epoch=15.ckpt` as a Pytorch Lightning checkpoint file. The following script evaluates its energy efficiency (EE) on the test data.

    ```python gnn_testing.py trained_model/checkpoints/best_epoch=15.ckpt```
    
    The results are saved in folder `test_results`:
    - `ee_results.csv` contains the median EE loss of EEPC-GNN compared to the optimal APG method, for each scenario.
    - For each scenario, a figure shows the EE cumulative distribution function of EEPC-GNN and APG.

## Training Pipeline

To train an EEPC-GNN model from scratch, first follow instructions 1 and 2 in the quickstart section above. Then, the training can be launched with:

```python gnn_training.py```

The choice of datasets for training and validation, as well as the hyperparameters (layers, batch size, learning rate, number of epochs, etc.) are hardcoded in this script.

## Test Results

The table below summarizes the EE losses at median of EEPC-GNN compared to the optimal APG method. Two checkpoints are considered here:
- `trained_model/checkpoints/best_epoch=15.ckpt` is the best model obtained after 15 epochs. It achieves good performance over all train and test scenarios.
- `trained_model/checkpoints/last_epoch=100.ckpt` is the same model trained for 100 epochs. It overfits to the training scenarios (first four rows in the table) with degraded performance on all other test scenarios.

Note that the results here are slightly better than the ones shown in the paper. The model in the paper is trained for 25 epochs with a different seed.

|Number of APs|Number of UEs|best model<br>(epoch=15)|overfitted model<br>(epoch=100)|
|-----------------|-----------------|--------|---------|
|50|10|0.23%|**0.03%**|
|60|15|0.10%|**-0.01%**|
|75|25|0.07%|**0.00%**|
|100|30|0.01%|**-0.01%**|
|100|40|0.05%|**0.01%**|
|200|10|**0.75%**|1.74%|
|200|15|**0.39%**|1.24%|
|200|20|**0.20%**|0.97%|
|200|25|**0.13%**|0.75%|
|200|30|**0.11%**|0.32%|
|200|35|**0.12%**|0.24%|
|200|40|**0.15%**|0.31%|
|300|40|**0.26%**|0.51%|
|300|65|**0.49%**|1.25%|
|355|37|**0.28%**|0.52%|
|400|10|**1.19%**|2.38%|
|400|15|**0.82%**|1.94%|
|400|20|**0.50%**|1.54%|
|400|25|**0.36%**|1.26%|
|400|30|**0.30%**|0.69%|
|400|35|**0.30%**|0.57%|
|400|40|**0.34%**|0.62%|
|500|40|**0.54%**|0.85%|
|600|40|**0.61%**|0.93%|
|700|40|**0.67%**|0.99%|

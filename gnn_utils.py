'''
© 2025 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import torch

def computeEnergyEfficiency(n_aps, n_ues, gamma, beta, pilots, theta):
    """
    Compute EE and SE for a given instance, ensures power constraint is not
    violated.

    Args:
        n_aps (int): Number of APs
        n_ues (int): Number of UEs
        gamma (tensor): n_aps x n_ues array of gamma
        beta (tensor): n_aps x n_ues array of beta, the LSFC
        pilots (tensor): n_ues x n_ues array of pilots
        theta (tensor): n_aps x n_ues array of theta, the power coefficient 

    Returns:
        EE_true(tensor): calculated EE 
        SE_per_user(tensor): n_ues array of calculated SE for each user 
    """
    n_antennas = 1
    B = 20
    T=200   
    tau = n_ues
    myalpha=(1/0.4)*torch.ones(n_aps, device = gamma.device)
    P_fix=0
    P_tc=0.2*n_aps
    P_bt=0.25*1e-3*n_aps
    P_0=0.825*n_aps
    P_fix_bar=P_fix + n_antennas*P_tc + P_0
    noise_p = 10**((-203.975+10*torch.log10(
        torch.tensor(B*1e6, device = gamma.device))+9)/10)
    Pd = 1/noise_p
        
    pil_mat = torch.abs(torch.matmul(pilots.T, pilots))
    p_tx = Pd * n_antennas**2 * torch.sum(torch.sqrt(gamma) * theta, axis=0)**2
    gamma_mat = gamma / beta
    
    pil_pwr = pil_mat*gamma_mat[:, :, None]
    gamma_tilde_mat = pil_pwr*beta[:, :, None]
    first_term_mat = torch.sum((gamma_tilde_mat*theta[:, :, None])**2,
                               axis = 0)

    tr_0 = torch.sum(torch.matmul(beta.T, theta**2).T, axis = 0)
    tr_1 = torch.sum(first_term_mat[:,:],axis = 0)

    interference = Pd * n_antennas *(tr_0 + tr_1)

    ap_power = torch.linalg.norm(theta, dim=1, keepdim = True)**2
    power_violated_index = ap_power > 1    
    # indices where the power constraint is satisfied
    power_ok_index = ~power_violated_index
    scaling_power = power_violated_index*ap_power + power_ok_index
    scaling_power = scaling_power.expand(-1, n_ues)
    scaling_power = scaling_power.view(n_aps, n_ues) 
    # scaling power tensor   
    theta = theta/torch.sqrt(scaling_power)
    
    # Calculate SE per user
    SE_per_user = torch.log2(1 + p_tx / (interference + 1))

    SE = torch.sum(SE_per_user)

    power =  P_fix_bar + Pd * noise_p * n_antennas  * \
        torch.norm(torch.sqrt(myalpha[:,None])* theta, p='fro')**2
    EE = B * (1 - tau / T) * SE/power

    EE_true = 1/(1/EE + torch.sum(P_bt))

    return EE_true, SE_per_user

def _compute_batch_EE(batch, y_hat):
    """Compute the energy efficiency and spectral efficiecny for a batch

    Args:
        batch (tensor): batch contining [x, theta, pilots]
        y_hat (tensor): current prediction for theta

    Returns:
        EE (tesnor.float64): True EE for the batch
        EE_hat (tesnor.float64): Predicted EE for the batch
        SE_per_user (tesnor): True SE per user for the batch
        SE_per_user_hat (tesnor): Predicted SE per user for the batch
    """
    # Number of UEs and APs for each sample of the batch
    n_ues = batch['channel'].n_ues
    n_aps = batch['channel'].n_aps
    # Number of features for each sample of the batch
    n_features = n_ues*n_aps
    
    batch_size = (batch['channel'].n_ues).size(dim = 0)
    
    ip_mean = batch['channel'].input_mean
    ip_std = batch['channel'].input_std
    op_mean = batch['channel'].output_mean
    op_std = batch['channel'].output_std
    gm_mean = batch['channel'].gamma_mean
    gm_std = batch['channel'].gamma_std

    input_mean = ip_mean.repeat_interleave(n_features)
    input_std = ip_std.repeat_interleave(n_features)

    output_mean = op_mean.repeat_interleave(n_features)
    output_std = op_std.repeat_interleave(n_features)

    gamma_mean = gm_mean.repeat_interleave(n_features)
    gamma_std = gm_std.repeat_interleave(n_features)

    # Deprocess the input and output features
    x = 2**((batch['channel'].beta[:,0]*input_std) +  input_mean)
    g = 2**((batch['channel'].gamma[:,0]*gamma_std) +  gamma_mean)
    y = (2**((batch['channel'].y[:,0]*output_std) +  output_mean))
    y_hat = (2**((y_hat[:,0]*output_std) +  output_mean))
    p = batch['channel'].pilots

    # Compute the "ue index" and "ap index" for each element of x (same for y)
    index_ues = []
    ues_pt = 0
    for num_ues, num_aps in zip(n_ues, n_aps):
        index_ues.append(
            torch.arange(ues_pt, ues_pt+num_ues,
                         device=x.device).repeat_interleave(num_aps))
        ues_pt += num_ues
    index_ues = torch.cat(index_ues)
    index_aps = []
    aps_pt = 0
    for num_ues, num_aps in zip(n_ues, n_aps):
        index_aps.append(
            torch.arange(aps_pt, aps_pt+num_aps,
                         device=x.device).repeat(num_ues))
        aps_pt += num_aps
    index_aps = torch.cat(index_aps)
 
    cnt_prev = 0
    cnt_new = 0

    cnt_p_prev = 0
    cnt_p_new = 0

    for i in range(batch_size):
       n_features = n_aps[i]*n_ues[i]
       cnt_new += n_features
       cnt_p_new += n_ues[i]*n_ues[i]

       beta = x[cnt_prev:cnt_new]
       theta = y[cnt_prev:cnt_new]
       theta_hat = y_hat[cnt_prev:cnt_new]
       gamma = g[cnt_prev:cnt_new]
       pilots = p[cnt_p_prev:cnt_p_new]

       beta = torch.reshape(beta, (n_aps[i], n_ues[i]))
       theta = torch.reshape(theta, (n_aps[i], n_ues[i]))
       theta_hat = torch.reshape(theta_hat, (n_aps[i], n_ues[i]))
       gamma = torch.reshape(gamma, (n_aps[i], n_ues[i]))
       pilots = torch.reshape(pilots, (n_ues[i], n_ues[i]))
       
       cnt_prev += n_features
       cnt_p_prev += n_ues[i]*n_ues[i]
        
       EE, SE_per_user =  computeEnergyEfficiency(n_aps[i], n_ues[i], gamma,
                                                  beta, pilots, theta)
       EE_hat, SE_per_user_hat =  \
        computeEnergyEfficiency(n_aps[i], n_ues[i], gamma,
                                beta, pilots, theta_hat)

    return EE, EE_hat, SE_per_user, SE_per_user_hat

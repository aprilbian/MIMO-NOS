# This file implements the looped_kbest with per-layer sorting 

import numpy as np
import torch
from utils import *
from mimo_sparc import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## System settings
V = 4
M = 256
Nt = 4
Nr = 4
F = 8
D = int(2*F*Nt)
I = int(np.log2(M))

# crc check settings
crc_len = 11
crc_str = '111000100001'

k_tree = 64

# NN settings
NN_enc = 4*D
NN_dec = 4*D
Hid = 128
is_res = True
equ_alpha = 0
batch_size = 1024
train_snr = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = {
    'V' : V,
    'M' : M,
    'D' : D,
    'Nt' : Nt,
    'Nr' : Nr,
    'F' : F,
    'NN_enc' : NN_enc,
    'NN_dec' : NN_dec, 
    'Hid' : Hid,
    'IS_RES' : is_res,
    'train_snr' : train_snr,
    'device' : device,
    'crc_str' : crc_str
}

print(opt)

name = 'MIMO_MORE'+'_IS_RES_'+str(is_res)+'_V'+str(V) + '_M'+str(M)+'_Nt'+str(Nt)+'_F'+str(F)+\
    '_SNR'+str(train_snr)+'_alpha'+str(equ_alpha)+'_batchsz'+str(batch_size)

#name = "MIMO_IS_RES_True_V4_M256_Nt2_F16_SNR10_alpha0_batchsz1024"
#name = 'MIMO_IS_RES_True_V4_M256_Nt4_F8_SNR7_alpha0_batchsz1024'

info_bit = int(V*(np.log2(M)))

SNR_db = [i+3 for i in range(6)]
max_iter = 20000
thres = 64

save_path = name+'/epoch10000.pth'

model = MIMO_HDM_model(opt).to(opt['device'])
model.load_state_dict(torch.load(save_path))
model.eval()


def construct_table():
    '''maintain a lookup table in the memory'''
    table = torch.zeros((V, M, F*Nt),dtype=torch.cfloat).to(device)
    one_hots = np.eye(M)                                            # batch_size = 1
    one_hots = torch.from_numpy(one_hots).float().to(device)
    one_hots = one_hots.unsqueeze(0)

    for v in range(V):
        for m in range(M):
            enc_vm = model.enc[v](one_hots[:,m,:])
            enc_vm = model.normalize(enc_vm, pwr = D/V)
            trans_sig = enc_vm/np.sqrt(2)
            trans_sig = trans_sig.view(1, F, Nt, 2)
            trans_sig = torch.view_as_complex(trans_sig).view(-1)

            table[v,m,:] = trans_sig
    
    lookup = table.detach().cpu().numpy()

    return lookup
    
lookup_table = construct_table()

def simpfy_mul(H, cand):
    cand = cand.reshape(V, M, F, Nt)
    H_cand = np.dot(cand, H.T)              # (V,M,F,Nr)
    return H_cand.reshape((V,M,Nr*F))

def rm_repetition(metric, idx_2d, k_best):
    '''Select k_best candidates w/o repetition'''
    point_k = 0
    sel_idx = np.zeros((k_best,),dtype=int)
    for k in range(k_best):
        val_k = metric[point_k]
        sel_idx[k] = idx_2d[point_k]
        point_k += 1
        while np.abs(metric[point_k]-val_k)<1e-5:
            point_k += 1
    return sel_idx


def loop_kbest_v1(y, H, k_best, ITER=0):
    '''decode the first layer after we
    finish decoding the last layer'''

    order = np.zeros(V,dtype=int)-1
    #order = np.arange(V)                    # order or not
    metric = np.zeros(k_best)
    vec_cum = np.zeros((k_best,D))
    index = np.zeros((V,k_best),dtype=int)

    LUT = simpfy_mul(H, lookup_table)

    u = 0
    
    # Determine the order
    
    min_val = np.zeros(V)+np.inf
    for i in range(V):
        corr_i = np.real(-2*np.dot(LUT[i,:,:],y.conj())+np.diag(np.dot(LUT[i,:,:],LUT[i,:,:].T.conj())))#
        min_val[i] = np.amin(corr_i)
    order[u] = np.argmin(min_val)

    # from the root:
    cand_u = LUT[order[u],:,:]
    cand_value = np.real(-2*np.dot(cand_u, y.conj())+np.diag(np.dot(cand_u,cand_u.T.conj()))) #
    
    index[u,:] = cand_value.argsort()[0:k_best]
    vec_cum = cand_u[index[u,:],:]
    metric = cand_value[index[u,:]]

    # for the rest (1,V-1) layers
    for v in range(1,V):
        # order v
        min_val = np.zeros(V)+np.inf
        order_y = y - vec_cum[0,:]
        for i in range(V):
            if np.sum(i==order):
                pass
            else:
                corr_i = np.real(-2*np.dot(LUT[i,:,:],order_y.conj())+np.diag(np.dot(LUT[i,:,:],LUT[i,:,:].T.conj())))#
                min_val[i] = np.amin(corr_i)
        order[v] = np.argmin(min_val)

        cand_v = LUT[order[v],:,:]
        cur_value = np.zeros((k_best,M))
        corr1 = np.real(-2*np.dot(cand_v,y.conj())+np.diag(np.dot(cand_v,cand_v.T.conj())))#

        corr2 = np.real(2*np.dot(vec_cum, cand_v.conj().T))                   # (kbest, M)
        cur_value = corr1.reshape(1,M) + corr2 + metric.reshape(k_best,1)
        cur_value = cur_value.reshape(-1)                                     # (kbest*M)
        
        idx_2d = cur_value.argsort()[0:k_best]
        metric = cur_value[idx_2d]

        new_idx, old_idx = ind2sub(idx_2d, M)
        index[v,:] = new_idx
        index[0:v,:] = index[0:v,old_idx]

        vec_cum_next = vec_cum
        vec_cum_next = cand_v[new_idx,:] + vec_cum[old_idx,:]
        vec_cum = vec_cum_next
    
    origin_idx = index[np.argsort(order), :]
    origin_order = order.copy()
    
    # Starting to find better results
    for _ in range(ITER):
        cur_value = np.zeros((k_best,M))
        cur_order = order[0]
        cand_iter = LUT[cur_order,:,:]
        cur_vec = cand_iter[index[0,:],:]; cur_vec_cum = vec_cum-cur_vec
        bias_metric = np.real(-2*np.dot(cur_vec.conj(),y)+np.diag(np.dot(cur_vec.conj(),cur_vec.T))
            +2*np.diag(np.dot(cur_vec.conj(),cur_vec_cum.T)))
        update_metric = metric-bias_metric
        vec_cum = cur_vec_cum

        corr1 = np.real(-2*np.dot(cand_iter,y.conj())+np.diag(np.dot(cand_iter,cand_iter.T.conj())))
        corr2 = np.real(2*np.dot(cur_vec_cum, cand_iter.conj().T))                   # (kbest, M)
        cur_value = corr1.reshape(1,M) + corr2 + update_metric.reshape(k_best,1)
        cur_value = cur_value.reshape(-1)                                         # (kbest*M)
        
        metric, idx_2d = np.sort(cur_value), np.argsort(cur_value)
        idx_2d = rm_repetition(metric, idx_2d, k_best)
        metric = cur_value[idx_2d]

        new_idx, old_idx = ind2sub(idx_2d, M)
        # update the order and index
        new_order, new_index = order.copy(), index.copy()
        new_order[0:V-1] = order[1:]; new_order[V-1] = order[0]
        new_index[V-1,:], new_index[0:V-1,:] = new_idx, index[1:,old_idx]
        order, index = new_order, new_index

        # update vec_cum
        vec_cum = cand_iter[new_idx,:] + vec_cum[old_idx,:]

    index = index[np.argsort(order), :]

    return index, order, origin_idx, origin_order

for snr in SNR_db:
    perr = 0
    count = 0
    err_list = np.zeros((V,1))
    while perr<=thres:
        count = count+1
        with torch.no_grad():
            raw_bits, one_hots, idx = gen_data(1, opt)
            x = torch.from_numpy(one_hots).float().to(device)
            enc_sig = np.zeros((F,Nt), dtype=complex)
            for v in range(V):
                enc_sig += lookup_table[v, idx[0,v], :].reshape(F,Nt)

            # MIMO channel
            H = (np.random.randn(Nr,Nt)+1j*np.random.randn(Nr,Nt))/np.sqrt(2)
            faded_sig = np.dot(enc_sig,H.T)                        # (F,Nr)
            faded_sig = faded_sig.reshape(-1)                      # D

            ## awgn channel
            N0 = 10**(snr/10)
            n = (np.random.randn(Nr*F)+1j*np.random.randn(Nr*F))/np.sqrt(2*N0/Nt)
            y = faded_sig + n

            #idx_list_s, order_s = dec_kbest(y,H,k_tree,idx)
            idx_list, order, origin_idx, origin_order = loop_kbest_v1(y,H,k_tree,V)
            _, dec_b = kbest_crc(idx_list, opt)

            if not np.array_equal(raw_bits,dec_b):
                perr += 1
                if(perr%20==0): print(perr)

    print('SNR = ', snr)
    print(perr/count)
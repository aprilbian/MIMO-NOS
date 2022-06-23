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
    'Nt' : 2,
    'Nr' : 2,
    'F' : 16,
    'NN_enc' : NN_enc,
    'NN_dec' : NN_dec, 
    'Hid' : Hid,
    'IS_RES' : is_res,
    'train_snr' : train_snr,
    'device' : device,
    'crc_str' : crc_str
}

print(opt)

name = 'MIMO'+'_IS_RES_'+str(is_res)+'_V'+str(V) + '_M'+str(M)+'_Nt'+str(Nt)+'_F'+str(F)+\
    '_SNR'+str(train_snr)+'_alpha'+str(equ_alpha)+'_batchsz'+str(batch_size)

name = "MIMO_IS_RES_True_V4_M256_Nt2_F16_SNR10_alpha0_batchsz1024"
#name = 'MIMO_IS_RES_True_V4_M256_Nt4_F8_SNR7_alpha0_batchsz1024'

info_bit = int(V*(np.log2(M)))

SNR_db = [i+3 for i in range(4)]
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

def update_step(index, order):
    '''update the index & order at each step'''
    idx_cp, order_cp = index.copy(), order.copy()
    idx_cp[0:V-1,:],order_cp[0:V-1,:] = index[1:,:], order[1:,:]
    order_cp[V-1,:] = order[0,:]
    return idx_cp, order_cp

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


def loop_kbest_v2(y, H, k_best, ITER=0):
    '''decode the first layer after we
    finish decoding the last layer'''

    order = np.zeros((V,k_best),dtype=int)-1
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
    order[u,:] = np.argmin(min_val)

    # from the root:
    cand_u = LUT[order[u,0],:,:]
    cand_value = np.real(-2*np.dot(cand_u, y.conj())+np.diag(np.dot(cand_u,cand_u.T.conj()))) #
    
    index[u,:] = cand_value.argsort()[0:k_best]
    vec_cum = cand_u[index[u,:],:]
    metric = cand_value[index[u,:]]

    # for the rest (1,V-1) layers
    for v in range(1,V):
        cur_value = np.zeros((k_best,M))
        for k in range(k_best):
            # order v
            
            min_val = np.zeros(V)+np.inf
            order_y = y - vec_cum[k,:]
            for i in range(V):
                if np.sum(i==order[:,k]):
                    pass
                else:
                    corr_i = np.real(-2*np.dot(LUT[i,:,:],order_y.conj())+np.diag(np.dot(LUT[i,:,:],LUT[i,:,:].T.conj())))
                    min_val[i] = np.amin(corr_i)
            order[v,k] = np.argmin(min_val)

            cand_v = LUT[order[v,k],:,:]
            
            corr1 = np.real(-2*np.dot(cand_v,y.conj())+np.diag(np.dot(cand_v,cand_v.T.conj())))
            corr2 = np.real(2*np.dot(vec_cum[k,:], cand_v.conj().T))
            cur_value[k,:] = np.squeeze(corr1) + np.squeeze(corr2) + metric[k]

        cur_value = cur_value.reshape(-1)                                     # (kbest*M)
        idx_2d = cur_value.argsort()[0:k_best]
        metric = cur_value[idx_2d]

        new_idx, old_idx = ind2sub(idx_2d, M)
        index[v,:] = new_idx
        index[0:v,:] = index[0:v,old_idx]
        order = order[:,old_idx]

        vec_cum_next = vec_cum.copy()
        for k in range(k_best):
            vec_cum_next[k,:] = LUT[order[v,k],new_idx[k],:] + vec_cum[old_idx[k],:]
        vec_cum = vec_cum_next

    # record the results w/o revisions...
    origin_index, origin_order = index.copy(), order.copy()
    for k in range(k_best):
        origin_index[:,k] = origin_index[np.argsort(origin_order[:,k]), k]
    
    # Starting to find better results
    for _ in range(ITER):
        cur_value = np.zeros((k_best,M))
        for k in range(k_best):
            # cancel out the previously decoded 1st layer
            idx_k = index[0,k]; order_k = order[0,k]; vec_k = LUT[order_k,idx_k,:]
            vec_cum_cur = vec_cum[k,:] - vec_k
            bias_metric = np.real(-2*np.dot(vec_k.conj(),y)+np.dot(vec_k.conj(),vec_k)+2*np.dot(vec_k.conj(),vec_cum_cur))
            metric_k = metric[k] - bias_metric
        
            # then start to decode the 1st layer 2nd time
            cand_v = LUT[order_k,:,:]
                
            corr1 = np.real(-2*np.dot(cand_v,y.conj())+np.diag(np.dot(cand_v,cand_v.T.conj())))
            corr2 = np.real(2*np.dot(vec_cum_cur, cand_v.conj().T))
            cur_value[k,:] = np.squeeze(corr1) + np.squeeze(corr2) + metric_k
            vec_cum[k,:] = vec_cum_cur
        
        cur_value = cur_value.reshape(-1)                                     # (kbest*M)
        metric, idx_2d = np.sort(cur_value), np.argsort(cur_value)
        idx_2d = rm_repetition(metric, idx_2d, k_best)
        metric = cur_value[idx_2d]
        #idx_2d = cur_value.argsort()[0:k_best]
        #metric = cur_value[idx_2d]

        new_idx, old_idx = ind2sub(idx_2d, M)

        # handling the index and the order
        index, order = update_step(index, order)
        index[V-1,:] = new_idx
        index[0:V-1,:] = index[0:V-1,old_idx]
        order = order[:,old_idx]

        # handling the vec_cum
        vec_cum_next = vec_cum.copy()
        for k in range(k_best):
            vec_cum_next[k,:] = LUT[order[V-1,k],new_idx[k],:] + vec_cum[old_idx[k],:]
        vec_cum = vec_cum_next

    for k in range(k_best):
        index[:,k] = index[np.argsort(order[:,k]), k]

    return index, order, origin_index, origin_order

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
            idx_list, order, origin_idx, origin_order = loop_kbest_v2(y,H,k_tree,V)
            _, dec_b = kbest_crc(idx_list, opt)

            if not np.array_equal(raw_bits,dec_b):
                perr += 1
                if(perr%20==0): print(perr)

    print('SNR = ', snr)
    print(perr/count)
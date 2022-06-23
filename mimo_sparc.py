import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F

class MIMO_HDM_model(nn.Module):

    def __init__(self,opt):
        super(MIMO_HDM_model, self).__init__()

        self.V = opt['V']
        self.M = opt['M']
        self.D = opt['D']

        self.Nt = opt['Nt']
        self.Nr = opt['Nr']
        self.F  = opt['F']

        self.NN_enc = opt['NN_enc']
        self.NN_dec = opt['NN_dec']
        self.Hid = opt['Hid']
        self.IS_RES = opt['IS_RES']
        self.device = opt['device']

        enc_list, dec_list, equ_list = [],[],[]
        for _ in range(self.V):

            enc = [nn.Linear(self.M,self.NN_enc),nn.ReLU(),nn.BatchNorm1d(self.NN_enc),
                    nn.Linear(self.NN_enc,self.NN_enc),nn.ReLU(),nn.BatchNorm1d(self.NN_enc),
                        nn.Linear(self.NN_enc,self.D)] 
            enc_list.append(nn.Sequential(*enc))
        self.enc = nn.ModuleList(enc_list)
        
        for _ in range(self.V):

            dec = [nn.Linear(self.D,self.NN_dec),nn.ReLU(),nn.BatchNorm1d(self.NN_dec),
                    nn.Linear(self.NN_dec,self.NN_dec),nn.ReLU(),nn.BatchNorm1d(self.NN_dec),
                        nn.Linear(self.NN_dec,self.M)]
            dec_list.append(nn.Sequential(*dec))
        self.dec = nn.ModuleList(dec_list)

        if self.IS_RES:
            res_equ1 = [nn.Linear(2*self.Nr+2*self.Nt*self.Nr, self.Hid), nn.ReLU(),
                            nn.Linear(self.Hid, self.Hid), nn.ReLU(),
                                nn.Linear(self.Hid, 2*self.Nt)]
            res_equ2 = [nn.Linear(4*self.Nt, self.Hid), nn.ReLU(),nn.Linear(self.Hid, 2*self.Nt)]
            #equ_list.append(nn.Sequential(*res_equ1)); equ_list.append(nn.Sequential(*res_equ2))
            #self.equ = nn.ModuleList(equ_list)
            self.res_equ1 = nn.Sequential(*res_equ1); self.res_equ2 = nn.Sequential(*res_equ2)
    
    def encoder(self, one_hots):

        batch_size = one_hots.shape[0]
        enc_sig = torch.zeros([batch_size,self.D]).to(self.device)

        for v in range(self.V):
            x_v = one_hots[:,v,:]
            enc_v = self.enc[v](x_v)

            enc_v = self.normalize(enc_v, self.D/self.V)
            enc_sig = enc_sig + enc_v

        trans_sig = enc_sig/np.sqrt(2)

        # map the trans_sig to the OFDM subcarriers and antennas
        trans_sig = trans_sig.view(batch_size, self.F, self.Nt, 2)
        trans_sig = torch.view_as_complex(trans_sig)

        return trans_sig

    def normalize(self, x, pwr=1):
        '''Normalization function'''
        power = torch.sum(x**2, -1, True)
        alpha = np.sqrt(pwr)/torch.sqrt(power)
        return alpha*x
    
    def MIMO_channel(self, batch):
        '''The MIMO channel, assumed quasi-static'''
        H = torch.randn((batch, self.Nr, self.Nt), dtype=torch.cfloat).to(self.device)
        extend_H = H.unsqueeze(1).repeat(1,self.F,1,1)   # (batch, F, Nr, Nt)
        self.H = extend_H


    def pass_channel(self, sig, snr):
        '''Passing through the MIMO-OFDM channel/freq domain'''
        mini_batch = sig.shape[0]
        sig = sig.unsqueeze(-1).view(mini_batch*self.F, self.Nt, 1)

        #### Define the MIMO channel
        self.MIMO_channel(mini_batch)      # self.H

        # AWGN   -- revise the channel
        sig_power = self.Nt
        noise_power = np.sqrt(sig_power*snr)
        noise = noise_power*torch.randn((mini_batch, self.F, self.Nr), dtype=torch.cfloat).to(self.device)

        #### Received signal
        extend_H = self.H.view(mini_batch*self.F, self.Nr, self.Nt)

        rec = torch.bmm(extend_H, sig).squeeze()              # (batchF, Nr)
        rec = rec.view(mini_batch, self.F, self.Nr)           # (batch, F, Nr)

        y = rec + noise

        return y

    def MMSE_Equ(self, y, snr):
        '''The MMSE equalization'''
        batch = y.shape[0]; batchF = batch*self.F
        y = (y.contiguous()).view(-1, self.Nr, 1)

        eye = (torch.eye(self.Nt, dtype=torch.cfloat).to(self.device)).repeat(batchF, 1, 1)
        eye = self.Nt*snr*eye
        #eye = snr*eye

        H = (self.H.contiguous()).view(-1, self.Nr, self.Nt)  # (batch*F, Nr, Nt)

        RHH = torch.bmm(torch.conj(H.permute(0,2,1)), H)
        Hy = torch.bmm(torch.conj(H.permute(0,2,1)), y)       # (batch*F, Nt, 1)

        inv_Ryy = torch.inverse(RHH + eye)

        x_equ = torch.bmm(inv_Ryy, Hy).squeeze()

        x_equ = (x_equ.contiguous()).view(-1, self.F, self.Nt)
        x_equ = (torch.view_as_real(x_equ).contiguous()).view((batch, self.F, 2*self.Nt))      # (batch, F, 2*Nt)

        return x_equ

    def RES_EQU(self, Y, x_equ):
        '''The Residual Assisted Equalization'''
        mini_batch = Y.shape[0]                 # y:(batch, F, Nr)
        Y = (torch.view_as_real(Y).contiguous()).view((mini_batch, self.F, 2*self.Nr))

        H = (torch.view_as_real(self.H).contiguous()).view((mini_batch, self.F, 2*self.Nr*self.Nt))
        HY = torch.cat((H,Y), dim=-1)
        residual = self.res_equ1(HY)           # (batch, F, 2Nt)
        
        #cat_vec = torch.cat((residual, x_equ), dim=-1)
        #residual = self.res_equ2(cat_vec)

        return residual          # (batch, F, 2Nt)

    def forward(self, x, SNR):
        mini_batch = x.shape[0]

        enc_sig = self.encoder(x)
        snr = 10**(-SNR/10)
        y = self.pass_channel(enc_sig, snr)     # (batch, F, Nr)
        x_equ1 = self.MMSE_Equ(y, snr)
        if self.IS_RES:
            residual = self.RES_EQU(y, x_equ1)
            x_equ = x_equ1 + residual
        
        x_equ = x_equ.view((mini_batch, 2*self.F*self.Nt))
        x_equ1 = x_equ1.view((mini_batch, 2*self.F*self.Nt))

        P_ = torch.zeros((mini_batch, self.V, self.M)).to(self.device)
        for v in range(self.V):
            
            p_v = self.dec[v](x_equ)
            P_[:,v,:] = p_v

        return P_, y, x_equ, x_equ1, torch.view_as_real(enc_sig).view(mini_batch,-1)
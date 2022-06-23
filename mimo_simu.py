import numpy as np 
import torch 
import torch.nn as nn
from torch.nn import init
from mimo_sparc import *
from visdom import Visdom
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

train_snr = 10

# NN settings
NN_enc = 4*D
NN_dec = 4*D
Hid = 128
is_res = True

# Training settings
lr = 2e-4
Epoches = 10001
Epoches = 0
batch_size = 1024
equ_alpha = 0

name = 'MIMO_MORE'+'_IS_RES_'+str(is_res)+'_V'+str(V) + '_M'+str(M)+'_Nt'+str(Nt)+'_F'+str(F)+\
    '_SNR'+str(train_snr)+'_alpha'+str(equ_alpha)+'_batchsz'+str(batch_size)

if not os.path.exists(name):
    os.mkdir(name)

viz_name = 'Test'+ name
viz = Visdom(env = viz_name, port = 8097)
viz.line([0.], [0.], win="train loss", opts=dict(title='train_loss'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_num = 75000
test_num = 10000
test_num = 800000

save_path = name+'/epoch10000.pth'
log_path = name + '/log_loss.txt'
test_path = name + '/log_per.txt'
loss_path = name + '/loss.npy'
f_log = open(log_path, "a+")
f_test = open(test_path, "a+")

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
    'device': device,
}

print(opt)

de2bin_map = torch.tensor([[0], [1]])
for i in range(I-1):
    de2bin_map_top = de2bin_map.clone()
    de2bin_map_down = de2bin_map.clone()

    de2bin_map_top = torch.cat((torch.zeros(2**(i+1), 1), de2bin_map_top), 1)
    de2bin_map_down = torch.cat((torch.ones(2**(i+1), 1), de2bin_map_down), 1)
    
    de2bin_map = torch.cat((de2bin_map_top, de2bin_map_down), 0)

de2bin_map = de2bin_map.numpy()

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=2e-2)

bin_vec = np.zeros((I,1), dtype= int)
for i in range(I):
    bin_vec[i, 0] = 2**(I-i-1)

def gen_bindata_vec(num):
    '''vectorize the calculatioin'''
    raw_bits = np.random.randint(2, size = (num, V, I))
    one_hots = np.zeros((num, V, M))

    idx = np.zeros((num, V))
    idx = np.dot(raw_bits, bin_vec)[:,:,0]   # (num, V)

    one_hots[np.arange(num)[:,None], np.arange(V)[None,:], idx] = 1

    return raw_bits, one_hots, idx

def loss_fun(prob, label, x_equ, s):

    loss = torch.zeros(1).to(opt['device'])
    for v in range(V):
        target_v = torch.argmax(label[:,v,:], dim = -1).long()
        loss = loss + nn.CrossEntropyLoss()(prob[:,v,:],target_v)
    loss = loss + equ_alpha*nn.MSELoss()(x_equ, s)
    
    return loss

model = MIMO_HDM_model(opt = opt).to(opt['device'])
model.apply(initNetParams)
model.load_state_dict(torch.load(save_path))

def test_model(epoch):
    '''Test the model every 1000 epochs'''
    model.eval()

    test_SNR = [4+4*i for i in range(4)]
    #test_SNR = [15]
    ber_list = []  

    test_bits, testset, _ = gen_bindata_vec(test_num)
    testset = torch.from_numpy(testset).float()

    with torch.no_grad():

        f_test.write('epoch is '+ str(epoch) + '\n')

        for snr in test_SNR:
            Iter = int(test_num/batch_size)
            berr = 0

            for iter in range(Iter):
                raw_data = testset[iter*batch_size:(iter+1)*batch_size,:,:].to(opt['device'])
                bit_data = test_bits[iter*batch_size:(iter+1)*batch_size,:,:]
                prob,_,_,_,_ = model(raw_data, snr) # (batch, V, M)

                # hard decision
                pred = torch.argmax(prob,dim = -1).cpu()
                cur_err = np.sum(abs(bit_data-de2bin_map[pred]))
                berr += cur_err

            ber = berr/(Iter*V*batch_size*I)
            ber_list.append(ber)

            print('for snr = ', snr, ' dB:')
            print('BER = ', ber)
            
            f_test.write('for snr = '+ str(snr) + ' dB:\n')
            f_test.write('BER = ' + str(ber) + '\n')

    print(ber_list)
    return ber_list[-1]  ## visualize BER @ -2 dB

if __name__ == '__main__':

    ## training......
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.4,verbose=1,min_lr=1e-6,patience=40)

    # For visualization
    viz_epoch = []
    viz_result = []
    
    for epoch in range(Epoches):

        _, trainset, idx = gen_bindata_vec(train_num)
        trainset = torch.from_numpy(trainset).float()
        idx = torch.from_numpy(idx).long()

        #adjust_lr(optimizer,epoch)
        Iter = int(train_num/batch_size)

        train_loss = 0.0

        for iter in range(Iter):
            raw_data = trainset[iter*batch_size:(iter+1)*batch_size,:,:].to(opt['device'])
            raw_idx = idx[iter*batch_size:(iter+1)*batch_size,:].to(opt['device'])

            prob,_,x_equ,s,_ = model(raw_data, opt['train_snr'])
            #prob,_,x_equ,s = model(raw_data, 30)

            optimizer.zero_grad()

            loss = loss_fun(prob, raw_data,x_equ,s)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()


        train_loss = train_loss/Iter
        scheduler.step(train_loss)    #update the learning rate


        if epoch % 10 == 0:
            print('epoch is', epoch)
            print('loss:', train_loss)
            #print('alpha:', model.alpha)

            # then test the current model
            ber = test_model(epoch)
            model.train()      # reset the model -- 'TRAIN'

            viz_epoch.append(epoch)
            viz_result.append([train_loss*0.1,np.log10(ber)])

            viz.line(
                    X=viz_epoch,
                    Y=viz_result,
                    opts={
                        'title': viz_name + ' loss over time',
                        'legend': [viz_name+ ' LOSS',viz_name+ ' BER'],
                        'xlabel': 'epoch',
                        'ylabel': 'loss'},
                    win='monitoring')
            
            f_log.write('epoch is '+str(epoch)+'\n')
            f_log.write('loss is ' +str(train_loss)+'\n')
        
        if epoch % 1000 ==0:
            save_epoch = name + '/epoch' + str(epoch) + '.pth'
            print('save the model at epoch ' + str(epoch) +'\n')
            torch.save(model.state_dict(), save_epoch)

    test_model(0)
    f_log.close()
    f_test.close()
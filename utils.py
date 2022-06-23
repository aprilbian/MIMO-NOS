import numpy as np 
import torch

def bi2str(raw_bits):
    '''bit sequence 2 string'''
    Len = raw_bits.shape[0]

    bits_list = [str(raw_bits[i]) for i in range(Len)]
    str_bits = ''.join(bits_list)

    return str_bits

def str2bi(str_bits):
    '''string 2 bit sequence(numpy)'''
    Len = len(str_bits)

    bit_seq = list(str_bits)
    bit_seq = [int(bit_seq[i]) for i in range(Len)]

    return np.array(bit_seq)

def ind2sub(id_2d, k_best):
    '''Python version of Matlab function: ind2sub'''

    Len = id_2d.shape[0]
    id_new, id_old = np.zeros(Len,dtype = int), np.zeros(Len,dtype = int)

    for idx in range(Len):

        id_new[idx] = id_2d[idx] % k_best
        id_old[idx] = int(id_2d[idx]/ k_best)
    
    return id_new, id_old

'''
def ind2sub(id_2d, keepnode, k_best):
    #Python version of Matlab function: ind2sub

    Len = id_2d.shape[0]
    id_new, id_old = np.zeros(Len,dtype = int), np.zeros(Len,dtype = int)

    for idx in range(Len):

        id_new[idx] = id_2d[idx] % k_best
        id_old[idx] = int(id_2d[idx]/ k_best)
    
    return id_old, id_new
'''

def de2bi(num,K):

    bit_seq = [0 for _ in range(K)]
    for i in range(K):
        s = num%2
        num = num//2
        bit_seq[K-i-1] = s 
        if num == 0:
            break
    return bit_seq

def bi2de(array):
    k = array.shape[0]

    num = 0
    for j in range(k):
        num += array[j]*2**(k-j-1)

    return num

def map_fun(I):
    de2bin_map = torch.tensor([[0], [1]])
    for i in range(I-1):
        de2bin_map_top = de2bin_map.clone()
        de2bin_map_down = de2bin_map.clone()

        de2bin_map_top = torch.cat((torch.zeros(2**(i+1), 1), de2bin_map_top), 1)
        de2bin_map_down = torch.cat((torch.ones(2**(i+1), 1), de2bin_map_down), 1)
        
        de2bin_map = torch.cat((de2bin_map_top, de2bin_map_down), 0)

    de2bin_map = de2bin_map.numpy()
    return de2bin_map



############# CRC Part #############


def XOR(str1, str2):
    ans = ''
    if str1[0] == '0':
        return '0', str1[1:]
    else:
        for i in range(len(str1)):
            if (str1[i] == '0' and str2[i] == '0'):
                ans = ans + '0'
            elif (str1[i] == '1' and str2[i] == '1'):
                ans = ans + '0'
            else:
                ans = ans + '1'
    return '1', ans[1:]
                

def CRC_Encoding(str1,str2):
    lenght = len(str2)
    str3 = str1 + '0'*(lenght-1)
    ans = ''
    yus = str3[0:lenght]
    for i in range(len(str1)):
        str4,yus = XOR(yus, str2)
        ans = ans+str4
        if i == len(str1)-1:
            break
        else:
            yus = yus+str3[i+lenght]
    ans = str1 + yus
    return ans

def CRC_Decoding(str1,str2): 
    lenght = len(str2)
    str3 = str1 + '0'*(lenght-1)
    ans = ''
    yus = str3[0:lenght]
    for i in range(len(str1)):
        str4,yus = XOR(yus, str2)
        ans = ans+str4
        if i == len(str1)-1:
            break
        else:
            yus = yus+str3[i+lenght]
    return yus == '0'*len(yus)


def kbest_crc(id_list, opt):
    '''Traverse the candidate list (from k-best) for crc'''
    k_best = id_list.shape[1]
    crc_pass = False
    test_bits = np.zeros((opt['V'],int(np.log2(opt['M']))))
    alter_solu = np.zeros((opt['V'],int(np.log2(opt['M']))))   # if crc fails, use the 1st element of id_list

    for k in range(k_best):
        candidate = id_list[:, k]

        for v in range(opt['V']):
            m_bit = de2bi(candidate[v], int(np.log2(opt['M']))) 
            test_bits[v,:] = m_bit
        
        if k==0:
            alter_solu = test_bits
        # crc
        crc_bit = test_bits.reshape(-1)
        crc_bit = crc_bit.astype(int)

        str_bit = bi2str(crc_bit)
        crc_pass = CRC_Decoding(str_bit, opt['crc_str'])

        if crc_pass:
            break
    
    if crc_pass:
        return crc_pass, test_bits
    else:
        return crc_pass, alter_solu


def gen_data(num, opt):
    '''generate data for training and testing'''
    crc_str = opt['crc_str']; crc_len = len(crc_str)-1
    V = opt['V']; M = opt['M']
    info_bit = int(V*np.log2(M))
    raw_bits = np.random.randint(2, size = info_bit - crc_len)
    str_bits = bi2str(raw_bits)

    str_codeword = CRC_Encoding(str_bits, crc_str)

    raw_bits = str2bi(str_codeword)
    raw_bits = raw_bits.reshape((num, V,-1))

    one_hots = np.zeros((num, V, M))
    idx = np.zeros((num,V),dtype=int)

    for n in range(num):
        for v in range(V):
            de_val = bi2de(raw_bits[n,v,:])
            idx[n,v] = de_val
            one_hots[n,v,de_val] = 1

    return raw_bits.squeeze(), one_hots, idx

if __name__ == "__main__":

    raw_bits = "10011000000000000000011"
    crc8 = "100000111"
    codeword = CRC_Encoding(raw_bits,crc8)
    print(codeword)
    code_list = list(codeword)
    code_list[10] = str((1+int(code_list[10]))%2)
    codeword = ''.join(code_list)
    print(CRC_Decoding(codeword,crc8))
clear all
tic;
rng(2021);

qpskMod = comm.QPSKModulator('BitInput',true);

L = 8;
bit_len_polar = 64; 
D = 128;
D_t = 64;
Nt = 4; Nr = 4;
F = 16;
crcLen = 11;
crc_poly = '11';

nMax = 10;
iTL = false;
iBL = true;

EbN0 = [3:1:8];
SNR = EbN0 + 10*log10(bit_len_polar/D_t);
thres = 10;

% Preparations for the ML detection
all_bits = de2bi([0:255],8,'left-msb');
all_syms = qpskMod(reshape(all_bits',2*Nt*2^(2*Nt),1));
all_syms = reshape(all_syms,Nt,2^(2*Nt)).';

sym_mappings = construct_table(2*Nt, all_bits, all_syms);

N_sim = [1e1 1e1 2e1 2e1 4e1 4e1 1e2 1e2 2e2 2e2 4e2 4e2]*2e2;
numRun_max = 1e4;


for r = 1:length(SNR)

    N0 = 10^(-SNR(r)/10);

    perr_polar = 0;
    
    n = 0;

    while (perr_polar < 64)
        n = n+1;
       
        bit_msg = randi([0 1], bit_len_polar-crcLen, 1);
        bit_info = nrCRCEncode(bit_msg,crc_poly);
        codeword = nrPolarEncode(bit_info, D,nMax,iTL);
        N = length(codeword);
        modIn = nrRateMatchPolar(codeword,bit_len_polar,D,iBL);
        x_tx = qpskMod(modIn);
        tx_sig = reshape(x_tx,Nt,F);
        
        H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);
        rx_sig = H*tx_sig;
        noise = sqrt(Nt*N0/2)*(randn(Nr,F) + 1j*randn(Nr,F));

        rx_sig = rx_sig + noise;
        % The ML detection
        LLR = zeros(D,1);
        for f = 1:F
            LLR(1+(f-1)*2*Nt:2*f*Nt) = ml_detection(rx_sig(:,f),H,sym_mappings,N0);
        end
        LLR(LLR>thres)=thres; LLR(LLR<-thres)=-thres;
        decIn = nrRateRecoverPolar(-LLR,bit_len_polar,N,iBL);
        bit_dec = nrPolarDecode(decIn,bit_len_polar,D,L,nMax,iTL,crcLen);
        perr_polar = perr_polar + ~isequal(bit_dec, bit_info);

        fprintf('\b\b\b\b%3g%%',min(round(100*n/N_sim(r)),100));
    end
    SNR(r)

    per_polar(r) = perr_polar/n
    toc
end
pathname = ['result',num2str(bit_len_polar),'.mat'];
save(pathname,'per_polar');


function sym_mappings = construct_table(num_bits, all_bits, all_syms)
    num_syms = floor(num_bits/2);
    sym_mappings = zeros(num_bits,2^num_bits,num_syms);
    half_comb = floor(2^num_bits/2);
    for n = 1:num_bits
        idx_n1 = (all_bits(:,n)==1); idx_n0 = (all_bits(:,n)==0);
        sym_n1 = all_syms(idx_n1,:); sym_n0 = all_syms(idx_n0,:);
        sym_mappings(n,1:half_comb,:) = sym_n1;
        sym_mappings(n,half_comb+1:end,:) = sym_n0;
    end
end

function soft_output = ml_detection(y,H,sym_mappings,N0)
    % ML detection with soft output
    Nt = size(H,1);
    num_bits = 2*Nt; num_comb = 2^num_bits;
    half_comb = floor(num_comb/2);
    soft_output = zeros(num_bits,1);
    for n=1:num_bits
        s1n = H*squeeze(sym_mappings(n,1:half_comb,:)).';
        s0n = H*squeeze(sym_mappings(n,half_comb+1:end,:)).';
        d1n = diag((y-s1n)'*(y-s1n)); d0n = diag((y-s0n)'*(y-s0n));
        llr1 = log(sum(exp(-1/(2*Nt*N0)*d1n))); llr0 = log(sum(exp(-1/(2*Nt*N0)*d0n)));
        soft_output(n) = llr1 - llr0;
        %soft_output(n) = -1/(2*Nt*N0)*(min(d1n)-min(d0n));
    end
end

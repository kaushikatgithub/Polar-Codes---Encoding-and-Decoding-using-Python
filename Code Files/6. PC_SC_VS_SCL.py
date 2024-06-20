"""
    Topic: Polar Codes
    Sub-topic: Polar Code SC vs SCL Decoding
"""
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import random as rd
from scipy.special import erfc

from Exports import RS as Reliability_Sequence
from Exports import getUSequence
from Exports import encode, decode_sc, decode_scl
from Exports import minsum, g
from Exports import PolarTransfromG2
from Exports import BPSK, AWGN
from Exports import Frozen

def sc_vs_scl(N, K, M):

    n = int(mt.log2(N))

    # Extracting Reliability Sequence for N bits
    rs = np.array([rs_pos for rs_pos in Reliability_Sequence if rs_pos < N])
    isFrozen = Frozen(rs, N, K)

    EbNodB_values = np.arange(0, 10, 0.5)
    P_success_sc = []
    P_success_scl = []
    BER_simulated_sc = []
    BER_simulated_scl = []
    BER_Theoretical = []

    # BER_Theoretical =  qfunc(np.sqrt(2 * Rate * EbNodB_values))
        
    for EbNodB in EbNodB_values:
        
        Success_sc = 0
        Success_scl = 0
        N_sim = 5000
        N_bit_errors_sc = 0
        N_bit_errors_scl = 0

        for sim in range(N_sim):

            Rate = K / N
            EbNo = 10 ** (EbNodB / 10)
            Sigma = mt.sqrt(1 / (Rate * EbNo))

            # Generating random K bits message
            message = np.random.randint(low = 0, high = 2, size = K, dtype = int)

            # Setting N-K frozen positions to zero and remaining K positions to message bits
            u_sequence = getUSequence(message, rs)
            encoded_message = encode(u_sequence, N)
            
            # Performing BPSK and AWGN
            bpsk_modulated_sequence = BPSK(encoded_message)
            received_sequqnce = AWGN(Sigma, bpsk_modulated_sequence)
            Likelihood = received_sequqnce
            enc_msg_while_decoding, decoded_u_sequence_sc = np.array(decode_sc(Likelihood, 0, n, isFrozen))
            decoded_u_sequence_scl = decode_scl(Likelihood, 0, n, M, isFrozen)[0][1]

            success_flag_sc = True
            success_flag_scl = True
            for i in range(N):
                if u_sequence[i] != decoded_u_sequence_sc[i]:
                    N_bit_errors_sc += 1
                    success_flag_sc = False
                if u_sequence[i] != decoded_u_sequence_scl[i]:
                    N_bit_errors_scl += 1
                    success_flag_scl = False
            if success_flag_sc:
                Success_sc += 1
            if success_flag_scl:
                Success_scl += 1

            # End for
        # End Nsim
        P_success_sc.append(Success_sc / N_sim)
        P_success_scl.append(Success_scl / N_sim)

        BER_simulated_sc.append(N_bit_errors_sc / (N * N_sim))
        BER_simulated_scl.append(N_bit_errors_scl / (N * N_sim))

        BER_Theoretical.append(0.5 * erfc(np.sqrt(Rate * EbNo)))

    plt.figure(1)
    plt.plot(EbNodB_values, np.array(BER_simulated_sc), label = f"N={N}, K={K} Sim BER SC") 
    plt.plot(EbNodB_values, np.array(BER_simulated_scl), label = f"N={N}, K={K} Sim BER SCL") 
    plt.plot(EbNodB_values, BER_Theoretical, label = f"N={N}, K={K} Theory BER", linestyle='--') 
    plt.ylim(1e-4, 1)
    # plt.xlim(0, 5)
    plt.yscale("log")
    plt.xlabel("Eb/No (dB)")  
    plt.ylabel("BER")  
    plt.title("BER v/s Eb/No SC vs SCL")
    plt.grid(True, which="both", axis="both")
    plt.legend()

    plt.figure(2)
    plt.plot(EbNodB_values, np.array(P_success_sc), label = f"N={N}, K={K} Psuccess SC") 
    plt.plot(EbNodB_values, np.array(P_success_scl), label = f"N={N}, K={K} Psuccess SCL") 
    plt.xlabel("Eb/No (dB)")  
    plt.ylabel("P_Success")  
    plt.title("P_Success v/s Eb/No SC vs SCL")
    plt.grid(True)
    plt.legend()

    plt.show()

# N: Total number of bits
# K: Number of message bits
# M: Max list size
sc_vs_scl(N = 8, K = 4, M = 4)

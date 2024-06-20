"""
    Topic: Polar Codes
    Sub-topic: Polar Code Successive Cancellation List Decoding, Bit Error Rate and Block Error Rate
"""
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import random as rd
from scipy.stats import norm
from scipy.special import erfc

from Exports import RS as Reliability_Sequence
from Exports import getUSequence
from Exports import encode, decode_scl as decode
from Exports import minsum, g
from Exports import PolarTransfromG2
from Exports import BPSK, AWGN
from Exports import Frozen

# Polar Codes
def polar_code(N, K, SCL_N):
    
    # N: Total number of bits
    # K: Number of message bits
    n = int(mt.log2(N))
    
    # Extracting Reliability Sequence for N bits
    rs = np.array([rs_pos for rs_pos in Reliability_Sequence if rs_pos < N])
    isFrozen = Frozen(rs, N, K)  

    EbNodB_values = np.arange(0, 10, 1)
    BER_simulated = []
    BLER = []
    BER_Theoretical = []
    P_success = []
    for EbNodB in EbNodB_values:

        Success = 0
        N_blk = 0
        N_sim = 1000
        N_bit_errors = 0
        
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

            # Decoding received message
            final_decisions = decode(Likelihood, 0, n, SCL_N, isFrozen)
            final_decisions.sort(key=lambda x: x[2])

            # Picking up the code with lowest path metric
            decoded_u_sequence = final_decisions[0][1]

            success_flag = True
            for i in range(N):
                if u_sequence[i] != decoded_u_sequence[i]:
                    N_bit_errors += 1
                    success_flag = False
            if success_flag:
                Success += 1
            else:
                N_blk += 1
            # End for
        # End Nsim

        P_success.append(Success / N_sim)
        BER_simulated.append(N_bit_errors / (N * N_sim))
        BLER.append(N_blk / (N_sim))
        BER_Theoretical.append(0.5 * erfc(np.sqrt(Rate * EbNo)))

    plt.figure(1)
    plt.plot(EbNodB_values, np.array(BER_simulated), label = f"N={N}, K={K}, M={SCL_N} Sim BER") 
    plt.plot(EbNodB_values, np.array(BLER), label = f"N={N}, K={K}, M={SCL_N} BLER") 
    plt.plot(EbNodB_values, BER_Theoretical, label = f"N={N}, K={K}, M={SCL_N} Theory BER") 
    plt.ylim(1e-4, 1)
    # plt.xlim(0, 5)
    plt.yscale("log")
    plt.xlabel("Eb/No (dB)")  
    plt.ylabel("BER")  
    plt.title("BER v/s SNR For SCL Decoder")
    plt.legend(fontsize=10)
    plt.grid(which="both", axis="both")

    plt.figure(2)
    plt.plot(EbNodB_values, np.array(P_success), label = f"N={N}, K={K}, M={SCL_N} p_success") 
    plt.xlabel("Eb/No (dB)")  
    plt.ylabel("P_Success")  
    plt.title("P_Success v/s Eb/No For SCL Decoder")
    plt.grid(True)
    plt.legend(fontsize=10)

    # End Monte Carlo
# End Polar Codes

# polar_code(N=4, K=2, SCL_N=4)
# polar_code(N=8, K=4, SCL_N=4)
polar_code(N=16, K=10, SCL_N=4)
# polar_code(N=64, K=40, SCL_N=4)
# polar_code(N=128, K=80, SCL_N=4)

plt.show()
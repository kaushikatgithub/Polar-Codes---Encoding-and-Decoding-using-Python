"""
    Topic: Polar Codes
    Sub-topic: Polar Code SCL VS SCL with Cyclic Redundancy Check
"""
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import random as rd
from scipy.special import erfc

from Exports import RS as Reliability_Sequence
from Exports import getUSequence
from Exports import encode, decode_scl as decode
from Exports import minsum, g
from Exports import PolarTransfromG2
from Exports import BPSK, AWGN
from Exports import Frozen

# CRC12: Telecom Systems
crc_list = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

def crc_encode(msg, crc_list):
    
    n, m = len(msg), len(crc_list)
    msg_with_crc = msg + [0] * (m-1)

    msg_poly = np.poly1d(msg_with_crc)
    crc_poly = np.poly1d(crc_list)

    q, remainder = np.polydiv(msg_poly, crc_poly)
    rem_bin = remainder.coefficients

    idx, rdx = n+m-2, len(rem_bin)-1
    while rdx >= 0:
        msg_with_crc[idx] = abs(int(int(rem_bin[rdx]) % 2))
        idx -= 1
        rdx -= 1
    # print(msg_with_crc)
    return list(msg_with_crc)

def crc_decode(decision_list, crc_list):

    valid = []
    invalid = []

    for decision in decision_list:
        msg = decision[1]
        msg_poly = np.poly1d(msg)
        crc_poly = np.poly1d(crc_list)

        q, remainder = np.polydiv(msg_poly, crc_poly)
        rem_bin = remainder.coefficients
        
        wrong = False
        for r in rem_bin:
            if abs(int(int(r) % 2)) != 0:
                wrong = True
                break
        if wrong:
            invalid.append(decision)
        else:
            valid.append(decision)
        
    if len(valid) != 0:
        valid.sort(key=lambda x: x[2])
        return valid[0]
    else:
        invalid.sort(key=lambda x: x[2])
        return invalid[0]
    
# Polar Codes
def polar_code(N, A, SCL_N):
    
    # N: Total number of bits
    # A: Number of message bits
    n = int(mt.log2(N))
    K = A + len(crc_list)-1
    # Extracting Reliability Sequence for N bits
    rs = np.array([rs_pos for rs_pos in Reliability_Sequence if rs_pos < N])
    isFrozen = Frozen(rs, N, K)  

    EbNodB_values = np.arange(0, 10, 0.5)
    P_success = []
    BER_simulated = []
    BER_Theoretical = []

    for EbNodB in EbNodB_values:

        N_sim = 1000
        N_bit_errors = 0
        Success = 0
        for sim in range(N_sim):

            Rate = A / N
            EbNo = 10 ** (EbNodB / 10)
            Sigma = mt.sqrt(1 / (2 * Rate * EbNo))

            # Generating random K bits message
            message = np.random.randint(low = 0, high = 2, size = A, dtype = int)

            # Adding CRC bits
            msg_with_crc = crc_encode(list(message), crc_list)

            # Setting N-K frozen positions to zero and remaining K positions to message bits
            u_sequence = getUSequence(msg_with_crc, rs)
            # print(u_sequence, end=" ")
            encoded_message = encode(u_sequence, N)
            
            # Performing BPSK and AWGN
            bpsk_modulated_sequence = BPSK(encoded_message)
            received_sequqnce = AWGN(Sigma, bpsk_modulated_sequence)
            Likelihood = received_sequqnce

            # Decoding received message
            final_decisions = decode(Likelihood, 0, n, SCL_N, isFrozen)
            final_decisions.sort(key=lambda x: x[2])

            # Picking up the code with lowest path metric
            decs = crc_decode(final_decisions, crc_list)
            decoded_u_sequence = decs[1]
            # print(decs)

            success_flag = True
            for i in range(N):
                if u_sequence[i] != decoded_u_sequence[i]:
                    N_bit_errors += 1
                    success_flag = False
            if success_flag:
                Success += 1
            
           
            # End for
        # End Nsim
        P_success.append(Success / N_sim)
        BER_simulated.append(N_bit_errors / (N * N_sim))
        BER_Theoretical.append(erfc(np.sqrt(Rate * EbNo)))
        print("EbNodB: ", EbNodB)
    plt.figure(1)
    plt.plot(EbNodB_values, np.array(BER_simulated), label = f"N={N}, K={A}, M={SCL_N} SimBER") 
    plt.plot(EbNodB_values, BER_Theoretical, label = f"N={N}, K={A} Theory BER") 
    plt.ylim(1e-4, 1)
    # plt.xlim(0, 5)
    plt.yscale("log")
    plt.xlabel("Eb/No (dB)")  
    plt.ylabel("BER")  
    plt.title("BER v/s SNR For SCL with CRC")
    plt.legend()
    plt.grid(which="both", axis="both")

    plt.figure(2)
    plt.plot(EbNodB_values, np.array(P_success), label = f"N={N}, K={A} Psuccess") 
    plt.xlabel("Eb/No (dB)")  
    plt.ylabel("P_Success")  
    plt.title("P_Success v/s Eb/No For SCL with CRC")
    plt.grid(True)
    plt.legend()
    # End Monte Carlo
# End Polar Codes

# polar_code(N=32, A=10, SCL_N=4)
polar_code(N=8, A=4, SCL_N=4)
# polar_code(N=16, A=1, SCL_N=4)
# polar_code(N=32, A=10, SCL_N=4)
# polar_code(N=64, K=40, SCL_N=4)
# polar_code(N=128, K=80, SCL_N=4)

plt.show()
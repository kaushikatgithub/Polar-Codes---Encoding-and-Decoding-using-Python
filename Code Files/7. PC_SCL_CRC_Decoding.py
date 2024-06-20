"""
    Topic: Polar Codes
    Sub-topic: Polar Code Successive Cancellation List Decoding with Cyclic Redundancy Check
"""
import numpy as np
import math as mt
import random as rd

from Exports import RS as Reliability_Sequence
from Exports import getUSequence
from Exports import encode, decode_scl as decode
from Exports import minsum, g
from Exports import PolarTransfromG2
from Exports import BPSK, AWGN
from Exports import Frozen, _isEqual

# CRC12: Telecom Systems
crc_list = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
# crc_list = [1, 1, 0, 0, 1]

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
        
# N: Total number of bits
# K: Number of message bits
N = 32
A = 10
SCL_N = 4

n = int(mt.log2(N))
EbNodB = 5
Rate = A / N
EbNo = 10 ** (EbNodB / 10)
Sigma = mt.sqrt(1 / (2 * Rate * EbNo))

# Extracting Reliability Sequence for N bits
rs = np.array([rs_pos for rs_pos in Reliability_Sequence if rs_pos < N])
print(f"Reliability sequence for N = {N}: ", end="")
print(rs)

# Generating random K bits message
# message = np.random.randint(low = 0, high = 2, size = A, dtype = int)
message = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
print("Message sequence: ", end="")
print(message)

# Adding CRC bits
msg_with_crc = crc_encode(list(message), crc_list)
K = len(msg_with_crc)

# Setting N-K frozen positions to zero and remaining K positions to message bits
u_sequence = getUSequence(msg_with_crc, rs)
# print(len(u_sequence))
print("U-Sequence: ", end="")
print(u_sequence)

encoded_message = encode(u_sequence, N)
print("Encoded message: ", end="")
print(encoded_message)

isFrozen = Frozen(rs, N, K)

# Performing BPSK and AWGN
bpsk_modulated_sequence = BPSK(encoded_message)
received_sequqnce = AWGN(Sigma, bpsk_modulated_sequence)
Likelihood = received_sequqnce

# Decoding received message
final_decisions = decode(Likelihood, 0, n, SCL_N, isFrozen)
# print("all decisions:", end="")
# for decision in final_decisions:
#     print(decision)

print("\ncrc decison: ", end="\n")
crc_decision = crc_decode(final_decisions, crc_list)
print(" Decoded Message: ", crc_decision[0])
print(" Decoded U-sequence: ", crc_decision[1])
print(" Path Metric: ", crc_decision[2])

if _isEqual(crc_decision[1], u_sequence):
    print("\nsuccessful decoding!")
else:
    print("\nError")

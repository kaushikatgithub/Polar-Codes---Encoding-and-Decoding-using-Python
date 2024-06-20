"""
    Topic: Polar Codes
    Sub-topic: Polar Code Successive Cancellation List Decoding
"""
import numpy as np
import math as mt

from Exports import RS as Reliability_Sequence
from Exports import getUSequence
from Exports import encode
from Exports import minsum, g
from Exports import PolarTransfromG2
from Exports import BPSK, AWGN
from Exports import Frozen, __isEqual

# N: Total number of bits
# K: Number of message bits
N = 8
K = 4
SCL_N = 4

n = int(mt.log2(N))
EbNodB = 5
Rate = K / N
EbNo = 10 ** (EbNodB / 10)
Sigma = mt.sqrt(1 / (Rate * EbNo))

# Extracting Reliability Sequence for N bits
rs = np.array([rs_pos for rs_pos in Reliability_Sequence if rs_pos < N])
print(f"Reliability sequence for N = {N}: ", end="")
print(rs)

# Generating random K bits message
message = np.random.randint(low = 0, high = 2, size = K, dtype = int)
print("Message sequence: ", end="")
print(message)

# Setting N-K frozen positions to zero and remaining K positions to message bits
u_sequence = getUSequence(message, rs)
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

def decode(L, node, depth, SCL_N, isFrozen):

    # We are at leaf node
    if depth == 0:

        decisions = []
        # If given bit is frozen then simply return 0
        if isFrozen[node]:
            if L[0] < 0:
                decisions.append([[0], [0], abs(L[0])])
            else:
                decisions.append([[0], [0], 0])
        else:
            # If L >= 0 then make hard decision that bit is 0 else it is 1
            if L[0] >= 0:
                decisions.append([[0], [0], 0])
                decisions.append([[1], [1], abs(L[0])])
            else:
                decisions.append([[1], [1], 0])
                decisions.append([[0], [0], abs(L[0])])
            # End hard decision
        return decisions
    # End leaf

    # L = [r1 r2]
    n = len(L)
    mid = n // 2
    r1 = L[0 : mid]
    r2 = L[mid : n]

    # Step L: Decode for the left child
    left_child_beliefs = minsum(r1, r2)
    left_child_decisions = decode(left_child_beliefs, 2*node, depth-1, SCL_N, isFrozen)

    final_decisions = []
    # Step R: Decode for the right child
    for lch_decision in left_child_decisions:

        right_child_beliefs = g(r1, r2, lch_decision[0])
        right_child_decisions = decode(right_child_beliefs, 2*node + 1, depth-1, SCL_N, isFrozen)

        # For every left combination perform Step R
        for rch_decision in right_child_decisions:

            # Step U: Combine left and right child decoded bits
            enc_msg = PolarTransfromG2(lch_decision[0], rch_decision[0])

            # Step End: Just combine decoded bits into a single list
            u_cap = []
            for u in lch_decision[1]:
                u_cap.append(u)

            for u in rch_decision[1]:
                u_cap.append(u)

            path_metric = lch_decision[2] + rch_decision[2]
            final_decisions.append([enc_msg, u_cap, path_metric])

    # Prunning the unnecessary branches
    final_decisions.sort(key=lambda x: x[2])
    final_decisions = sorted(final_decisions, key=lambda x: x[2])
    if len(final_decisions) > SCL_N:
        for i in range(SCL_N, len(final_decisions)):
            final_decisions.pop()
    
    return final_decisions
# End decoding

# Decoding received message
final_decisions = decode(Likelihood, 0, n, SCL_N, isFrozen)

print()
print("|    Dcoded Message     |   Decoded U-sequence   |           Path Metric            |")
for decision in final_decisions:
    print(decision[0], decision[1], decision[2], __isEqual(u_sequence, decision[1]), sep=" ")
print()




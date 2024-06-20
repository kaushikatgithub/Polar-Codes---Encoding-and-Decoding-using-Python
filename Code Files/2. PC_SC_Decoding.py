"""
    Topic: Polar Codes
    Sub-topic: Polar Code Successive Cancellation Decoding
"""
import numpy as np
import math as mt

from Exports import RS as Reliability_Sequence
from Exports import getUSequence
from Exports import encode
from Exports import minsum, g
from Exports import PolarTransfromG2
from Exports import BPSK, AWGN
from Exports import Frozen, _isEqual

# N: Total number of bits
# K: Number of message bits
N = 8
K = 3
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

encoded_message = encode(u_sequence)
print("Encoded message: ", end="")
print(encoded_message)

isFrozen = Frozen(rs, N, K)

# Performing BPSK and AWGN
bpsk_modulated_sequence = BPSK(encoded_message)
received_sequqnce = AWGN(Sigma, bpsk_modulated_sequence)
Likelihood = received_sequqnce
# print(Likelihood)

# Time Complexity: NLogN
def decode(L, node, depth, isFrozen):

    # We are at leaf node
    if depth == 0:

        # If given bit is frozen then simply return 0
        if isFrozen[node]:
            return [[0], [0]]
        else:
            # If L >= 0 then make hard decision that bit is 0 else it is 1
            if L[0] >= 0:
                return [[0], [0]]
            else:
                return [[1], [1]]
            # End hard decision
    # End leaf

    # L = [r1 r2]
    n = len(L)
    mid = n // 2
    r1 = L[0 : mid]
    r2 = L[mid : n]

    # Step L: Decode for the left child
    left_child_beliefs = minsum(r1, r2)
    u_cap_combined_left, u_cap_left = decode(left_child_beliefs, 2*node, depth-1, isFrozen)

    # Step R: Decode for the right child
    right_child_beliefs = g(r1, r2, u_cap_combined_left)
    u_cap_combined_right, u_cap_right = decode(right_child_beliefs, 2*node + 1, depth-1, isFrozen)

    # Step U: Combine left and right child decoded bits
    enc_msg = PolarTransfromG2(u_cap_combined_left, u_cap_combined_right)
    
    # Step End: Just combine decoded bits into a single list
    u_cap = []
    for u in u_cap_left:
        u_cap.append(u)

    for u in u_cap_right:
        u_cap.append(u)

    return [enc_msg, u_cap]
# End decoding

# Decoding received message
enc_msg, decoded_message = np.array(decode(Likelihood, 0, n, isFrozen))

print("Encoded message generate vie decoding process: ", end="")
print(enc_msg)
print("Decoded U-sequence: ", end="")
print(decoded_message)

if _isEqual(u_sequence, decoded_message):
    print("Successfully decoded the message!")
else:
    print("Erroneous message!")


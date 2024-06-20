"""
    Topic: Polar Codes
    Sub-topic: Polar Code Encoding using Iterative and Recursive Method
"""

import numpy as np
import math as mt

from Exports import RS as Reliability_Sequence
from Exports import getUSequence
from Exports import PolarTransfromG2

# N: Total number of bits
# K: Number of message bits
N = 16
K = 8
n = int(mt.log2(N))

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

# Polar Code Encoding using iterative method
def encode_iterative(u_sequence, N):

    encoded_message = np.copy(u_sequence)
    m = 1
    for depth in range(n-1, -1, -1):
        for idx in range(0, N, 2*m):

            u0 = encoded_message[idx: idx+m]
            u1 = encoded_message[idx+m: idx+2*m]

            # u0 = [u0 + u1, u1]
            encoded_message[idx: idx+m] = (u0 + u1) % 2
            encoded_message[idx+m: idx+2*m] = u1
        # End
        m = m * 2
    # End
    return encoded_message

# Time Complexity NLogN
# Polar Code Encoding using recursive method
def encode_recursive(u_sequence):

    n = len(u_sequence)
    if n == 1:
        return u_sequence
    
    u0 = u_sequence[0:n//2]
    u1 = u_sequence[n//2:]

    encoded_u0 = encode_recursive(u0)
    encoded_u1 = encode_recursive(u1)
    
    encoded_message = PolarTransfromG2(encoded_u0, encoded_u1)
    
    return encoded_message

encoded_sequnce_rec = np.array(encode_recursive(u_sequence))
print("Encoded sequence recursive: ", end="")
print(encoded_sequnce_rec)

encoded_sequnce_it = encode_iterative(u_sequence, N)
print("Encoded sequence iterative: ", end="")
print(encoded_sequnce_it)
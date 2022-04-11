"""
Main.py file.

Authors
_______

Thomas Dormart
Enzo Benoit-Jeannin
"""
# Import statements
import sys
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy...

del sys.modules["numpy"].fft  # ... except FFT helpers

# [TODO: Deliverable 1] A generalized discrete Fourier transform
# inSignal - 1D (sampled) input signal numpy array
# s - sign parameter with default value -1 for the DFT vs. iDFT setting
# returns the DFT of the input signal
def DFT(inSignal, s: int = -1):
    y = np.zeros(inSignal.shape, dtype=complex)
    ### BEGIN SOLUTION
    # Solution is based on the given formula in the assignment instructions.

    N = inSignal.shape[0]  # N is the length of the input

    # Create a matrix with each cell storing the product of its indices
    # We need this matrix because in the formula f_hat = M @ f
    # both the index of the row and the column are needed to compute the value in M
    # But these two indices are multiplied, so we can create this matrix
    ab = np.arange(N) * np.arange(N).reshape((N, 1))

    # Use the formula given for M and exp(jx) = cos(x)+jsin(x)
    M = np.cos(s * 2 * np.pi * ab / N) + np.sin(s * 2 * np.pi * ab / N) * 1j

    # Obtain the result using f_hat = M @ f
    y = inSignal @ M
    ### END SOLUTION
    return y


# [TODO: Deliverable 2] The inverse DFT, relying on your generalized DFT routine above
# inSignal - complex-valued (sampled) 1D DFT input numpy array
# returns the iDFT of the input signal
def iDFT(inSignal: complex):
    N = inSignal.shape[0]  # N is the length of the input
    # The iDFT is the inverse discrete fourier transform
    # and it's just the DFT of the same signal, with a change of sign for s
    # multiplied by 1 / N
    # This method avoids code duplication
    return 1 / N * DFT(inSignal, 1)


# [TODO: Deliverable 3] A generalized 2D discrete Fourier transform routine
# inSignal2D - 2D (sampled) input signal numpy array
# s - sign parameter with default value -1 for the DFT vs. iDFT setting
# returns the 2D DFT of the input signal
def DFT2D(inSignal2D, s: int = -1):
    y = np.zeros(inSignal2D.shape, dtype=complex)
    ### BEGIN SOLUTION
    # This solution is based ont he given formula in the assignment instructions

    N = inSignal2D.shape[0]  # N is the length of the input

    # Create a matrix with each cell storing the product of its indices
    # We need this matrix because in the formula f_hat = M @ f
    # both the index of the row and the column are needed to compute the value in M
    # But these two indices are multiplied, so we can create this matrix
    ab = np.arange(N) * np.arange(N).reshape((N, 1))

    # Use the same formula we used for the 1D Discrete Fourier Transform to compute M
    M = np.cos(s * 2 * np.pi * ab / N) + np.sin(s * 2 * np.pi * ab / N) * 1j

    # Use the given formula to obatin the 2D Dicrete Fourier Transform
    y = M @ (M @ inSignal2D.T).T
    ### END SOLUTION
    return y


# [TODO: Deliverable 4] The inverse 2D DFT, relying on your 2D DFT routine
# inSignal2D - complex-valued (sampled) 2D DFT input array
# returns the 2D iDFT of the input signal
def iDFT2D(inSignal2D: complex):
    N = inSignal2D.shape[0]  # N is the length of the input
    # The iDFT2D is the 2D inverse discrete fourier transform
    # and it's just the 2D DFT of the same signal, with a change of sign for s
    # multiplied by 1 / N^2
    # This method avoids code duplication
    return (1 / N ** 2) * DFT2D(inSignal2D, 1)

if __name__ == "__main__":
    print("Test main.py")
    print("test Thomas")
    print("Salut Thomas")

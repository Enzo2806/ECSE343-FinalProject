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
import timeit

def DFT(inSignal, s: int = -1):
    """
    :param inSignal: 1D (sampled) input signal numpy array
    :param s: sign parameter with default value -1 for the DFT vs. iDFT setting
    :return: returns the DFT of the input signal
    """
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


def iDFT(inSignal: complex):
    """
    :param inSignal: complex-valued (sampled) 1D DFT input numpy array.
    :return: returns the iDFT of the input signal.
    """
    N = inSignal.shape[0]  # N is the length of the input
    # The iDFT is the inverse discrete fourier transform
    # and it's just the DFT of the same signal, with a change of sign for s
    # multiplied by 1 / N
    # This method avoids code duplication
    return 1 / N * DFT(inSignal, 1)


def DFT2D(inSignal2D, s: int = -1):
    """
    :param inSignal2D: complex-valued (sampled) 2D DFT input array.
    :param s: sign parameter with default value -1 for the DFT vs. iDFT setting.
    :return: the generated DFT2D given the input signal.
    """
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


def iDFT2D(inSignal2D: complex):
    """
    :param inSignal2D: complex-valued (sampled) 2D DFT input array.
    :return: the generated iDFT2D given the input signal.
    """
    N = inSignal2D.shape[0]  # N is the length of the input
    # The iDFT2D is the 2D inverse discrete fourier transform
    # and it's just the 2D DFT of the same signal, with a change of sign for s
    # multiplied by 1 / N^2
    # This method avoids code duplication
    return (1 / N ** 2) * DFT2D(inSignal2D, 1)


def FFT_CT(inSignal, s: int = -1):
    """
    Function generating the FFT of the given signal using the Cooley-Tukey ALgorithm
    :param inSignal: 1D (sampled) input signal numpy array
    :param s: sign parameter with default value -1 for the DFT vs. iDFT setting
    :return: returns the DFT of the input signal
    """
    result = np.zeros(inSignal.shape, dtype=complex)

    N = inSignal.shape[0]

    if N == 0:
        raise ValueError("Invalid signal: length 0")

    if N == 1:
        result = inSignal
    else:
        w = np.exp(complex(0, s * 2 * np.pi / N))
        diag_e_a = np.diag(np.full(int(N / 2), w) ** np.arange(N / 2))

        inSignal_e_hat = FFT_CT(inSignal[::2])
        inSignal_o_hat = FFT_CT(inSignal[1::2])

        result[0: int(N / 2)] = inSignal_e_hat + diag_e_a @ inSignal_o_hat
        result[int(N / 2): N] = inSignal_e_hat - diag_e_a @ inSignal_o_hat

    return result


if __name__ == "__main__":
    # Test the FFT_CT function
    print("FIRST TEST")
    print("__________________________")
    print("Description: Calls the FFT_CT function with a signal of length 0.")
    print("Expected Output: SHould throw the following Exception: \"Invalid signal: length 0\" ")
    print("Output:")
    try:
        testVector = np.random.rand(0)
        FFT_CT(testVector)
    except Exception as e:
        print(e)

    print("SECOND TEST")
    print("__________________________")
    print("Description: ")
    print("Expected Output: SHould throw the following Exception: \"Invalid signal: length 0\" ")
    print("Output:")

    #TODO: Implement a shit ton of other tests here.

    #TODO: BENCHMARKSS
    # for N in range (2 ** 45):
    signal = np.random.rand(2**14)
    start_time = timeit.default_timer()
    FFT_CT(signal)
    print("Time taken:")
    print(timeit.default_timer() - start_time)
    start_time = timeit.default_timer()
    np.fft.fft(signal)
    print(timeit.default_timer() - start_time)
    start_time = timeit.default_timer()
    DFT(signal)
    print(timeit.default_timer() - start_time)


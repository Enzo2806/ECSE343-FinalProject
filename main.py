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
from PIL import Image

def DFT(inSignal, s: int = -1):
    """
    Function to generate the discrete Fourier transform of the input signal.
    :param inSignal: 1D (sampled) input signal numpy array
    :param s: sign parameter with default value -1 for the DFT vs. iDFT setting
    :return: returns the DFT of the input signal
    """
    y = np.zeros(inSignal.shape, dtype=complex)
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
    return y


def iDFT(inSignal):
    """
    Function generating the inverse DFT, relying on the generalized DFT routine above.
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
    Function to generate the 2-Dimensional discrete Fourier transform of the input signal.
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


def iDFT2D(inSignal2D):
    """
    Function to generate the inverse 2-Dimensional discrete Fourier transform of the input signal.
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


<<<<<<< Updated upstream
def iFFT_CT(inSignal):
    """
    Function generating the inverse FFT of the given signal using the Cooley-Tukey ALgorithm
    :param inSignal: complex-valued (sampled) 1D DFT input numpy array
    :return: the iFFT of the input signal
    """
    N = inSignal.shape[0]  # N is the length of the input
    # The iFFT is the inverse fast fourier transform
    # and it's just the FFT of the same signal, with a change of sign for s
    # multiplied by 1 / N
    return 1 / N * FFT_CT(inSignal, 1)


def FFT_CT2D(inSignal2D, s: int = -1):
    """
    Function generating the 2-Dimensional FFT of the given 2D signal using the Cooley-Tukey Algorithm we implemented
    :param inSignal2D: 2D (sampled) input signal numpy array
    :param s: sign parameter with default value -1 for the FFT vs. iFFT setting
    :return: the 2D DFT of the input signal
    """
    return np.transpose(FFT_CT(np.transpose(FFT_CT(inSignal2D, s)), s))


def iFFT_CT2D(inSignal2D):
    """
    Function generating the inverse 2-Dimensional FFT of the given 2D signal using the Cooley-Tukey Algorithm we implemented
    :param inSignal2D: complex-valued (sampled) 2D DFT input array.
    :return: the generated iDFT2D given the input signal.
    """
    N = inSignal2D.shape[0]  # N is the length of the input
    # The iFFT2D is the 2D inverse fast fourier transform
    # and it's just the 2D FFT of the same signal, with a change of sign for s
    # multiplied by 1 / N^2
    return (1 / N ** 2) * FFT_CT2D(inSignal2D, 1)
=======
def FFT_CT_base(inSignal, k, s: int = -1):
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

    if N == (k):
        result = DFT(inSignal,s)
    else:
        w = np.exp(complex(0, s * 2 * np.pi / N))
        diag_e_a = np.diag(np.full(int(N / 2), w) ** np.arange(N / 2))

        inSignal_e_hat = FFT_CT_base(inSignal[::2],k)
        inSignal_o_hat = FFT_CT_base(inSignal[1::2],k)

        result[0: int(N / 2)] = inSignal_e_hat + diag_e_a @ inSignal_o_hat
        result[int(N / 2): N] = inSignal_e_hat - diag_e_a @ inSignal_o_hat

    return result
>>>>>>> Stashed changes


if __name__ == "__main__":
    # Test the FFT_CT function
    print("FIRST TEST")
    print("__________________________")
    print("Description: Calls the FFT_CT function with a signal of length 0.")
    print("Expected Output: Should throw the following Exception: \"Invalid signal: length 0\" ")
    print("Output:")
    try:
        testVector = np.random.rand(0)
        FFT_CT(testVector)
    except Exception as e:
        print(e)

<<<<<<< Updated upstream
    print("SECOND TEST")
    print("__________________________")
    print("Description: ")
    print("Expected Output: Should throw the following Exception: \"Invalid signal: length 0\" ")
    print("Output:")

    # TODO: Implement a shit ton of other tests here.

    # VALIDATION TESTS, should be performed on a wide range of values
    # TODO: Test FFT against our DFT , returns a boolean, TRUE if result is same, FALSE if result is different
    print("THIRD TEST")
=======
    #TODO: Implement a shit ton of other tests here.

    # VALIDATION TESTS, should be performed on a wide range of values
    #TODO: Test FFT against our DFT , returns a boolean, TRUE if result is same, FALSE if result is different
    print("\nSECOND TEST")
>>>>>>> Stashed changes
    print("__________________________")
    print(
        "Description: Should print TRUE if the the result found by our FFT is the same as the one computed using our DFT, else it prints FALSE")
    print("Output:")
    N = 2 ** 8
    signal = np.random.rand(N)
    fft = FFT_CT(signal)
    dft = DFT(signal)
    npfft = np.fft.fft(signal)
<<<<<<< Updated upstream
    print(np.allclose(fft, dft))
=======
    print("Is FFT_CT equal to DFT ?")
    print (np.allclose(fft, dft))
    print("Is FFT_CT close to np.fft.fft ?")
>>>>>>> Stashed changes
    print(np.allclose(fft, npfft))

    # TODO: Test FFT against ftt.fft, returns a boolean, TRUE if result is same, FALSE if result is different

    # TODO: Test 2DFFT
    testVector_2D = np.random.rand(2 ** 10, 2 ** 10)
    print("2D Test")
    print(np.allclose(FFT_CT2D(testVector_2D), np.fft.fft2(testVector_2D)))
    print(np.allclose(FFT_CT2D(testVector_2D), DFT2D(testVector_2D)))

<<<<<<< Updated upstream
    # TODO: BENCHMARKSS, should be performed on a wide range of values
    signal = np.random.rand(2 ** 8)
=======
    #TODO: Test FFT against ftt.fft, returns a boolean, TRUE if result is same, FALSE if result is different



    #TODO: BENCHMARKSS, should be performed on a wide range of values
    print("\nThird TEST")
    print("__________________________")
    print("Description: Should print the time taken to compute the Discrete Fourier Transform using different algorithm ")
    print("Output:")
    signal = np.random.rand(2**15)
>>>>>>> Stashed changes
    start_time = timeit.default_timer()
    FFT_CT(signal)
    print ("Time taken by FFT_CT:")
    print(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    np.fft.fft(signal)
    print("Time taken by np.fft.fft:")
    print(timeit.default_timer() - start_time)

    #start_time = timeit.default_timer()
    #DFT(signal)
    #print("Time taken by DFT")
    #print(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    FFT_CT_base(signal, 2**7)
    print("Time taken by FFT_CT_BASE:")
    print(timeit.default_timer() - start_time)

    bench_CT_result = {}
    bench_oldDFT_result = {}

    for N in (2 ** p for p in range(1, 13)):
        signal = np.random.rand(N)
        start_time = timeit.default_timer()
        FFT_CT(signal)
        bench_CT_result[N] = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        DFT(signal)
        bench_oldDFT_result[N] = timeit.default_timer() - start_time


    lists = sorted(bench_CT_result.items())  # sorted by key, return a list of tuples
    N, CT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, CT_time)

    lists = sorted(bench_oldDFT_result.items())  # sorted by key, return a list of tuples
    N, oldDFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, oldDFT_time)
    plt.title("Ma bite sur ton front")
    plt.xlabel("Value of N (Size of the array)")
    plt.ylabel("Time taken (seconds)")
    plt.show()

<<<<<<< Updated upstream
    # TODO: Application
    # image = np.asarray(Image.open('Koala.jpg'))
    # print(image)

    # Load test data from file
    image = None
    with open('A3-test-data.npy', 'rb') as f:
        image = np.load(f)
    print(image)

    cmap = plt.get_cmap('gray')
    _, plots = plt.subplots(2, 1, figsize=(10, 7))
    plt.setp(plots, xticks=[], yticks=[])
    plots[0].set_title('test input image', size=8)
    plots[0].imshow(image, cmap, vmin=0, vmax=1)
    plots[1].set_title('blurred output image', size=8)
    plots[1].imshow(FFT_CT2D(image), cmap, vmin=0, vmax=1)
    plt.show()  # this is a blocking call; kill the plotting window to continue execution
=======

    #print("\nSECOND TEST")
    #print("__________________________")
    #print("Description: ")
    #print("Expected Output: ")
    #print("Output:")


>>>>>>> Stashed changes

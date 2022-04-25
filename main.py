"""
Main.py file.

Authors
_______

Thomas Dormart
Enzo Benoit-Jeannin
"""
# Import statements
import os.path
import sys
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy...
import timeit
from PIL import Image, ImageOps


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

#@profile
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

        inSignal_e_hat = FFT_CT(inSignal[::2], s)
        inSignal_o_hat = FFT_CT(inSignal[1::2], s)

        result[0: int(N / 2)] = inSignal_e_hat + diag_e_a @ inSignal_o_hat
        result[int(N / 2): N] = inSignal_e_hat - diag_e_a @ inSignal_o_hat

    return result


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
    return FFT_CT(FFT_CT(inSignal2D, s).T, s).T

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
    return 1 / N ** 2 * FFT_CT2D(inSignal2D, 1)

#@profile
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
        result = DFT(inSignal, s)
    else:
        w = np.exp(complex(0, s * 2 * np.pi / N))
        diag_e_a = np.diag(np.full(int(N / 2), w) ** np.arange(N / 2))

        inSignal_e_hat = FFT_CT_base(inSignal[::2], k, s)
        inSignal_o_hat = FFT_CT_base(inSignal[1::2], k, s)

        #inSignal_e_hat = FFT_CT_base(inSignal[::2], k)
        #inSignal_o_hat = FFT_CT_base(inSignal[1::2], k)

        result[0: int(N / 2)] = inSignal_e_hat + diag_e_a @ inSignal_o_hat
        result[int(N / 2): N] = inSignal_e_hat - diag_e_a @ inSignal_o_hat

    return result

def iFFT_CT_base(inSignal):
    """
    Function generating the inverse FFT of the given signal using the Cooley-Tukey ALgorithm
    :param inSignal: complex-valued (sampled) 1D DFT input numpy array
    :return: the iFFT of the input signal
    """
    N = inSignal.shape[0]  # N is the length of the input
    # The iFFT is the inverse fast fourier transform
    # and it's just the FFT of the same signal, with a change of sign for s
    # multiplied by 1 / N
    return 1 / N * FFT_CT_base(inSignal,2**5,1)

def FFT_CT2D_base(inSignal2D,s: int = -1):
    """
    Function generating the 2-Dimensional FFT of the given 2D signal using the Cooley-Tukey Algorithm we implemented
    :param inSignal2D: 2D (sampled) input signal numpy array
    :param s: sign parameter with default value -1 for the FFT vs. iFFT setting
    :return: the 2D DFT of the input signal
    """
    return np.transpose(FFT_CT_base(np.transpose(FFT_CT_base(inSignal2D, 2**5,s)),2**5,s))

def compress(Image, p):
    """

    :param Image: 2D array representing an image
    :param p: percentage of compression
    :return: the compressed image
    """
    fft = FFT_CT2D(Image)
    fft_sorted = np.sort(np.abs(fft.reshape(-1)))
    thresh = fft_sorted[int(np.floor((1 - p) * len(fft_sorted)))]
    ind = np.abs(fft) > thresh
    return iFFT_CT2D(fft * ind).real


def compress_DFT(Image, p):
    """

    :param Image: 2D array representing an image
    :param p: percentage of compression
    :return: the compressed image
    """
    dft = DFT2D(Image)
    dft_sorted = np.sort(np.abs(dft.reshape(-1)))
    thresh = dft_sorted[int(np.floor((1 - p) * len(dft_sorted)))]
    ind = np.abs(dft) > thresh
    return iDFT2D(dft * ind).real
    

def first():
    print("Validation Tests")
    # Test the FFT_CT function
    print("\nFIRST TEST")
    print("__________________________")
    print("Description: Calls the FFT_CT function with a signal of length 0.")
    print("Expected Output: Should throw the following Exception: \"Invalid signal: length 0\" ")
    print("Output:")
    try:
        testVector = np.random.rand(0)
        FFT_CT(testVector)
    except Exception as e:
        print(e)
    print("__________________________")

def second():
    print("\nSECOND TEST")
    print("__________________________")
    print(
        "Description: Should print TRUE if the the result found by our FFT is the same as the one computed using our DFT and numpy's FFT, else it prints FALSE")
    print("The test is performed on a 1D 2^14 long array")
    print("Expected Output: Should return true for both methods")
    print("Output:")

    N = 2 ** 14
    signal = np.random.rand(N)
    fft = FFT_CT(signal)
    dft = DFT(signal)
    npfft = np.fft.fft(signal)
    fftbase = FFT_CT_base(signal, 2 ** 5,-1)

    print("Is FFT_CT equal to DFT ?")
    print(np.allclose(fft, dft))

    print("Is FFT_CT equal to numpy's FFT?")
    print(np.allclose(fft, npfft))

    print ("Is FFT_CT_base equal to FFT_CT?")
    print(np.allclose(fftbase,fft))

    print("Is FFT_CT_base equal to DFT?")
    print(np.allclose(fftbase,dft))

    print("Is FFT_CT_base equal to numpy's FFT?")
    print(np.allclose(fftbase,npfft))
    print("__________________________")

def third():
    print("\nTHIRD TEST")
    print("__________________________")
    print(
        "Description: Should print TRUE if the the result found by our FFT is the same as the one computed using our DFT, else it prints FALSE")
    print("The test is performed on a 2D 2^12 long array")
    print("Output:")
    testVector_2D = np.random.rand(2 ** 12, 2 ** 12)
    fft = FFT_CT2D(testVector_2D)
    npfft = np.fft.fft2(testVector_2D)
    dft = DFT2D(testVector_2D)
    #fftbase= FFT_CT2D_base(testVector_2D,-1)
    print("Is FFT_CT2D equal to numpy's 2D FFT?")
    print(np.allclose(fft, npfft))

    print("Is FFT_CT2D equal to DFT2D?")
    print(np.allclose(fft, dft))

    #print("Is FFT_CT2D_base equal to numpy's 2D FFT?")
    #print(np.allclose(fftbase, npfft))

    #print("Is FFT-CT2D_base equal to FFT_2D?")
    #print(np.allclose(fftbase,fft))
    #print(np.allclose(iFFT_CT2D(FFT_CT2D(testVector_2D)), testVector_2D))

    print("__________________________")

def fourth():
    print("\nFOURTH TEST")
    print("__________________________")
    print(
        "Description: Should print TRUE if the the result found by our inverse FFT is the same as the one computed using our inverse DFT and numpy's inverse FFT, else it prints FALSE")
    print("The test is performed on a 1D 2^14 long array")
    print("Expected Output: Should return true for both methods")
    print("Output:")

    N = 2 ** 14
    signal = np.random.rand(N)
    ifft = iFFT_CT(signal)
    idft = iDFT(signal)
    inpfft = np.fft.ifft(signal)
    ifftbase = iFFT_CT_base(signal)
    print("Is iFFT_CT equal to iDFT ?")
    print(np.allclose(ifft, idft))

    print("Is iFFT_CT equal to numpy's iFFT?")
    print(np.allclose(ifft, inpfft))

    print("Is iFFT_CT_base equal to iFFT_CT?")
    print(np.allclose(ifftbase,ifft))

    print("Is iFFT_CT_base equal to iDFT?")
    print(np.allclose(ifftbase,idft))
    print("__________________________")

def fifth():
    print("\nFIFTH TEST")
    print("__________________________")
    print(
        "Description: Should print TRUE if the the result found by our FFT is the same as the one computed using our DFT, else it prints FALSE")
    print("The test is performed on a 2D 2^12 long array")
    print("Output:")
    testVector_2D = np.random.rand(2 ** 12, 2 ** 12)
    ifft = iFFT_CT2D(testVector_2D)
    inpfft = np.fft.ifft2(testVector_2D)
    idft = iDFT2D(testVector_2D)
    # ifftbase= iFFT_CT2D_base(testVector_2D,-1)
    print("Is iFFT_CT2D equal to numpy's 2D iFFT?")
    print(np.allclose(ifft, inpfft))

    print("Is iFFT_CT2D equal to iDFT2D?")
    print(np.allclose(ifft, idft))

    # print("Is iFFT_CT2D_base equal to numpy's 2D iFFT?")
    # print(np.allclose(ifftbase, inpfft))

    # print("Is iFFT-CT2D_base equal to iFFT_2D?")
    # print(np.allclose(ifftbase,ifft))
    # print(np.allclose(iFFT_CT2D(FFT_CT2D(testVector_2D)), testVector_2D))

def sixth():
    print("\nSIXTH TEST")
    print("__________________________")
    print("Description: This test should determine which base case is the best for our recursion")
    print(
        "The test is performed on 1D array's ranging from 2^1 to 2^10, the average is made on testing each base case a thousand time."
        "From our graph, we expect the best base case to be between 2^0 and 2^9, so we will try base cases in this range"
    "This test might take several minutes to finish even on HEDT")
    print("Expected Output: Should return the average time to compute FFT with each base case")
    print("Output:")
    averagetimes = np.zeros(10)
    for base in range(9):
        length = base
        totalaverage = 0
        while length <= 10:  # power of 2 used for the length of the signal
            signal = signal = np.random.rand(2 ** length)
            i = 0
            average = 0
            while i <= 30:  # Iterates a thousand time over an array of the same base case and the length of array
                start_time = timeit.default_timer()
                FFT_CT_base(signal, 2 ** base)
                average = average + (timeit.default_timer() - start_time)
                i = i + 1
            average = float(average / (i - 1))
            totalaverage = totalaverage + average
            #print("For a signal a signal of length 2^" + str(length) + " and a base case of 2^" + str(base) + " the average time to do the computation is " + str(average) + " seconds")
            length = length + 1
        averagetimes[base] = totalaverage
    print(averagetimes)

def seventh():
    print("\nSEVENTH TEST")
    print("__________________________")
    print(
        "Description: Should print the time taken (in seconds) to compute the Discrete Fourier Transform using different algorithm "
    "This test can take several minutes to finish even on HEDT")
    print("Output:")
    averagefftct = 0
    averagefftbase = 0
    averagefftnp = 0
    averageDFT = 0
    for n in range(30):
        signal = np.random.rand(2 ** 15)

        start_time = timeit.default_timer()
        FFT_CT(signal)
        #print("Time taken by FFT_CT:")
        #print(timeit.default_timer() - start_time)
        averagefftct = averagefftct + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        np.fft.fft(signal)
        #print("Time taken by np.fft.fft:")
        #print(timeit.default_timer() - start_time)
        averagefftnp = averagefftnp + (timeit.default_timer() - start_time)

        #start_time = timeit.default_timer()
        #DFT(signal)
        #print("Time taken by DFT")
        #print(timeit.default_timer() - start_time)
        #averageDFT = averageDFT + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        FFT_CT_base(signal, 2 ** 5)
        #print("Time taken by FFT_CT_BASE:")
        #print(timeit.default_timer() - start_time)
        averagefftbase = averagefftbase + (timeit.default_timer() - start_time)
    averagefftct = averagefftct / 30
    averagefftbase = averagefftbase / 30
    averagefftnp = averagefftnp / 30
    averageDFT = averageDFT / 30
    print("Over 30 iterations")
    print("Average Time FFT CT: " + str(averagefftct))
    print("Average Time FFT BASE: " + str(averagefftbase))
    print("Average Time FFT NP: " + str(averagefftnp))
    print("Average Time DFT:" + str(averageDFT))
    print("__________________________")

def eighth():
    print("\nEIGHTH TEST")
    print("__________________________")
    print(
        "Description: Should print the average time taken (in seconds) to compute the 2D Discrete Fourier Transform using different algorithm ")
    print("Output:")
    averagefftct = 0
    averagefftbase = 0
    averagefftnp = 0
    averageDFT = 0
    for n in range(30):
        signal = np.random.rand(2 ** 12,2**12)

        start_time = timeit.default_timer()
        FFT_CT2D(signal)
        # print("Time taken by FFT_CT:")
        # print(timeit.default_timer() - start_time)
        averagefftct = averagefftct + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        np.fft.fft2(signal)
        # print("Time taken by np.fft.fft:")
        # print(timeit.default_timer() - start_time)
        averagefftnp = averagefftnp + (timeit.default_timer() - start_time)

        # start_time = timeit.default_timer()
        # DFT2D(signal)
        # print("Time taken by DFT")
        # print(timeit.default_timer() - start_time)
        # averageDFT = averageDFT + (timeit.default_timer() - start_time)

        #start_time = timeit.default_timer()
        #FFT_CT2D_base(signal, 2 ** 5)
        # print("Time taken by FFT_CT_BASE:")
        # print(timeit.default_timer() - start_time)
        #averagefftbase = averagefftbase + (timeit.default_timer() - start_time)
    averagefftct = averagefftct / 30
    averagefftbase = averagefftbase / 30
    averagefftnp = averagefftnp / 30
    averageDFT = averageDFT / 30
    print("Over 30 iterations")
    print("Average Time FFT CT2D: " + str(averagefftct))
    print("Average Time FFT BASE 2D: " + str(averagefftbase))
    print("Average Time FFT NP 2D: " + str(averagefftnp))
    print("Average Time DFT 2D:" + str(averageDFT))
    print("__________________________")

def ninth():
    print("\nNINTH TEST")
    print("__________________________")
    print(
        "Description: Should print the average time taken (in seconds) to compute the inverse Discrete Fourier Transform using different algorithm "
        "This test can take several minutes to finish even on HEDT")
    print("Output:")
    averagefftct = 0
    averagefftbase = 0
    averagefftnp = 0
    averageDFT = 0
    for n in range(30):
        signal = np.random.rand(2 ** 14)

        start_time = timeit.default_timer()
        iFFT_CT(signal)
        # print("Time taken by FFT_CT:")
        # print(timeit.default_timer() - start_time)
        averagefftct = averagefftct + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        np.fft.ifft(signal)
        # print("Time taken by np.fft.fft:")
        # print(timeit.default_timer() - start_time)
        averagefftnp = averagefftnp + (timeit.default_timer() - start_time)

        # start_time = timeit.default_timer()
        # DFT(signal)
        # print("Time taken by DFT")
        # print(timeit.default_timer() - start_time)
        # averageDFT = averageDFT + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        iFFT_CT_base(signal)
        # print("Time taken by FFT_CT_BASE:")
        # print(timeit.default_timer() - start_time)
        averagefftbase = averagefftbase + (timeit.default_timer() - start_time)
    averagefftct = averagefftct / 30
    averagefftbase = averagefftbase / 30
    averagefftnp = averagefftnp / 30
    averageDFT = averageDFT / 30
    print("Over 30 iterations")
    print("Average Time iFFT CT: " + str(averagefftct))
    print("Average Time iFFT BASE: " + str(averagefftbase))
    print("Average Time iFFT NP: " + str(averagefftnp))
    print("Average Time iDFT:" + str(averageDFT))
    print("__________________________")

def tenth():
    print("\nTENTH TEST")
    print("__________________________")
    print(
        "Description: Should print the average time taken (in seconds) to compute the 2D Discrete Fourier Transform using different algorithm ")
    print("Output:")
    averagefftct = 0
    averagefftbase = 0
    averagefftnp = 0
    averageDFT = 0
    for n in range(30):
        signal = np.random.rand(2 ** 12, 2 ** 12)

        start_time = timeit.default_timer()
        iFFT_CT2D(signal)
        # print("Time taken by FFT_CT:")
        # print(timeit.default_timer() - start_time)
        averagefftct = averagefftct + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        np.fft.ifft2(signal)
        # print("Time taken by np.fft.fft:")
        # print(timeit.default_timer() - start_time)
        averagefftnp = averagefftnp + (timeit.default_timer() - start_time)

        # start_time = timeit.default_timer()
        # DFT2D(signal)
        # print("Time taken by DFT")
        # print(timeit.default_timer() - start_time)
        # averageDFT = averageDFT + (timeit.default_timer() - start_time)

        # start_time = timeit.default_timer()
        # iFFT_CT2D_base(signal)
        # print("Time taken by FFT_CT_BASE:")
        # print(timeit.default_timer() - start_time)
        # averagefftbase = averagefftbase + (timeit.default_timer() - start_time)
    averagefftct = averagefftct / 30
    averagefftbase = averagefftbase / 30
    averagefftnp = averagefftnp / 30
    averageDFT = averageDFT / 30
    print("Over 30 iterations")
    print("Average Time iFFT CT2D: " + str(averagefftct))
    print("Average Time iFFT BASE 2D: " + str(averagefftbase))
    print("Average Time iFFT NP 2D: " + str(averagefftnp))
    print("Average Time iDFT 2D:" + str(averageDFT))
    print("__________________________")


def ourgraph():
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
    plt.plot(N, CT_time, label="FFT")

    lists = sorted(bench_oldDFT_result.items())  # sorted by key, return a list of tuples
    N, oldDFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, oldDFT_time, label="DFT")
    plt.title("Graph comparing the time efficiency of the DFT and FFT algorithm")
    plt.xlabel("Value of N (Size of the array)")
    plt.ylabel("Time taken (seconds)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #first()

    #second()

    #third()

    #fourth()

    #fifth()

    # sixth()

    # seventh()

    eighth()

    # ninth()

    tenth()

    #Create the graph
    # ourgraph()

    # Application
    # print("The application of the FFT algorithm we chose consists of compressing a given image to reduce its size.")
    # print("This is done by taking the fast fourier transform of an input image (in grayscale) and by removing "
    #       "(set to 0) the lowest frequencies of that fft depending on a threshold we specify. We then use the inverse "
    #       "fft algorithm we wrote to go generate the compressed image.")
    # print("We defined a function named \"compress\" that takes as argument an image and the percentage of frequencies "
    #       "to keep in the fast fourier transform. For instance, compress(imageTest, 0.7) will remove 30% of all the "
    #       "frequencies in the fourier domain (only the lowest ones).")
    #
    # print("\nFIRST APPLICATION TEST")
    # print("__________________________")
    # print("Description: This first application test will compress a koala image multiple times.")
    # print("Expected Output: The output should be a matplotlib plot, showing the input image and all 5 resulting "
    #       "compressed images with their respective new sizes.")
    # print("Output:")
    #
    # # https://e2eml.school/convert_rgb_to_grayscale.html
    # image = np.asarray(Image.open('Koala.jpg'))  # Import the image
    # grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    # plt.imsave('GreyKoala.jpg', grayscale)
    # size = os.path.getsize('GreyKoala.jpg') / 1000  # Get the size of the original image in kilobytes
    #
    # cmap = plt.get_cmap('gray')
    # _, plots = plt.subplots(3, 2, figsize=(10, 8))
    # plt.setp(plots, xticks=[], yticks=[])
    # plots[0][0].set_title("Input image, size={}KB".format(size), size=10)
    # plots[0][0].imshow(grayscale, cmap, vmin=0, vmax=1)
    # compression_percentage = [0.7, 0.2, 0.1, 0.05, 0.002]
    #
    # count1 = 1
    # count2 = 0
    #
    # for j in compression_percentage:
    #     compressed = compress(grayscale, j)
    #
    #     plt.imsave('compressed.jpg', compressed)  # Save the compressed image in the current package to get its size
    #     size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
    #     plots[count1][count2].set_title("Compression removed {}% of input image\nNew size={}KB".format(100-100*j, size), size=10)
    #     plots[count1][count2].imshow(compressed, cmap,  vmin=0, vmax=1)
    #     count1 += 1
    #     if count1 == 3:
    #         count1 = 0
    #         count2 += 1
    # plt.show()

    print("\nSECOND APPLICATION TEST")
    print("__________________________")
    print("Description: This second application test will compare the dft and fft algorithms in the compression"
          "of the same koala image.")
    print("Expected Output: Benchmark of 4 compressions of 4 different images comparing the DFT and FFT algorithms.")
    print("Output: ")

    # Import all four images following the same procedure as in the firts test.
    # Convert RGB Pictures to greyscale, save them to check their size (which changed since we converted them to
    # greyscale).
    image_VieuxLyon = np.asarray(Image.open('VieuxLyon.jpg'))  # Import the image
    grayscale_VieuxLyon = np.dot(image_VieuxLyon[..., :3], [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    plt.imsave('VieuxLyon.jpg', grayscale_VieuxLyon, cmap='gray')
    size_VieuxLyon = os.path.getsize('VieuxLyon.jpg') / 1000  # Get the size of the original image in kilobytes

    image_MoulinRouge = np.asarray(Image.open('MoulinRouge.jpg'))  # Import the image
    grayscale_MoulinRouge = np.dot(image_MoulinRouge[..., :3], [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    plt.imsave('MoulinRouge.jpg', grayscale_MoulinRouge, cmap='gray')
    size_MoulinRouge = os.path.getsize('MoulinRouge.jpg') / 1000  # Get the size of the original image in kilobytes

    image_Koala = np.asarray(Image.open('Koala.jpg'))  # Import the image
    grayscale_Koala = np.dot(image_Koala[..., :3], [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    plt.imsave('Koala.jpg', grayscale_Koala, cmap='gray')
    size_Koala = os.path.getsize('Koala.jpg') / 1000  # Get the size of the original image in kilobytes

    image_Fourviere = np.asarray(Image.open('Fourviere.jpg'))  # Import the image
    grayscale_Fourviere = np.dot(image_Fourviere[..., :3],
                             [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    plt.imsave('Fourviere.jpg', grayscale_Fourviere, cmap='gray')
    size_Fourviere = os.path.getsize('Fourviere.jpg') / 1000  # Get the size of the original image in kilobytes

    # Configure the plot by adding the input images in a first column.
    cmap = plt.get_cmap('gray')
    _, plots = plt.subplots(4, 2, figsize=(10, 8))
    plt.setp(plots, xticks=[], yticks=[])
    plots[0][0].set_title("Input 4096x4096 image, size={}KB".format(size_VieuxLyon), size=10)
    plots[0][0].imshow(grayscale_VieuxLyon, cmap, vmin=0, vmax=1)
    plots[1][0].set_title("Input 2048x2048 image, size={}KB".format(size_MoulinRouge), size=10)
    plots[1][0].imshow(grayscale_MoulinRouge, cmap, vmin=0, vmax=1)
    plots[2][0].set_title("Input 1024x1024 image, size={}KB".format(size_Koala), size=10)
    plots[2][0].imshow(grayscale_Koala, cmap, vmin=0, vmax=1)
    plots[3][0].set_title("Input 512x512 image, size={}KB".format(size_Fourviere), size=10)
    plots[3][0].imshow(grayscale_Fourviere, cmap, vmin=0, vmax=1)

    # Variables to keep track of the time taken for each compression for both DFT and FFT algorithm
    bench_FFT_result = {}
    bench_DFT_result = {}

    # 4096x4096 picture compression using FFT
    start_time = timeit.default_timer()
    compressed = compress(grayscale_VieuxLyon, 0.009)
    bench_FFT_result[4096] = timeit.default_timer() - start_time
    # Save the compressed image in the current package to get its size
    plt.imsave('compressed.jpg', compressed, cmap='gray')
    plots[0][1].set_title("Compression removed 99,1% of input image", size=10)
    plots[0][1].imshow(compressed, cmap, vmin=0, vmax=1)

    # 4096x4096 picture compression using DFT
    start_time = timeit.default_timer()
    compressed = compress_DFT(grayscale_VieuxLyon, 0.009)
    bench_DFT_result[4096] = timeit.default_timer() - start_time

    # 2048x2048 picture compression using FFT
    start_time = timeit.default_timer()
    compressed = compress(grayscale_MoulinRouge, 0.009)
    bench_FFT_result[2048] = timeit.default_timer() - start_time
    # Save the compressed image in the current package to get its size
    plt.imsave('compressed.jpg', compressed, cmap='gray')
    size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
    plots[1][1].imshow(compressed, cmap, vmin=0, vmax=1)

    # 2048x2048 picture compression using DFT
    start_time = timeit.default_timer()
    compressed = compress_DFT(grayscale_MoulinRouge, 0.009)
    bench_DFT_result[2048] = timeit.default_timer() - start_time

    # 1024x1024 picture compression using FFT
    start_time = timeit.default_timer()
    compressed = compress(grayscale_Koala, 0.009)
    bench_FFT_result[1024] = timeit.default_timer() - start_time
    # Save the compressed image in the current package to get its size
    plt.imsave('compressed.jpg', compressed, cmap='gray')
    size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
    plots[2][1].imshow(compressed, cmap, vmin=0, vmax=1)

    # 1024x1024 picture compression using DFT
    start_time = timeit.default_timer()
    compressed = compress_DFT(grayscale_Koala, 0.009)
    bench_DFT_result[1024] = timeit.default_timer() - start_time

    # 512x512 picture compression using FFT
    start_time = timeit.default_timer()
    compressed = compress(grayscale_Fourviere, 0.009)
    bench_FFT_result[512] = timeit.default_timer() - start_time
    # Save the compressed image in the current package to get its size
    plt.imsave('compressed.jpg', compressed, cmap='gray')
    size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
    plots[3][1].imshow(compressed, cmap, vmin=0, vmax=1)

    # 1024x1024 picture compression using DFT
    start_time = timeit.default_timer()
    compressed = compress_DFT(grayscale_Fourviere, 0.009)
    bench_DFT_result[512] = timeit.default_timer() - start_time
    plt.show()

    lists = sorted(bench_FFT_result.items())  # sorted by key, return a list of tuples
    N, CT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, CT_time, label="FFT")

    lists = sorted(bench_DFT_result.items())  # sorted by key, return a list of tuples
    N, oldDFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, oldDFT_time, label="DFT")

    plt.title("Graph comparing the time efficiency of the DFT and FFT algorithm for the compression application.")
    plt.xlabel("Value of N (Size of the image)")
    plt.ylabel("Time taken (seconds)")
    plt.legend()
    plt.show()

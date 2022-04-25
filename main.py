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
        inSignal_o_hat = FFT_CT_base(inSignal[1::2], s)

        result[0: int(N / 2)] = inSignal_e_hat + diag_e_a @ inSignal_o_hat
        result[int(N / 2): N] = inSignal_e_hat - diag_e_a @ inSignal_o_hat

    return result


def compress(Image, p):
    """

    :param Image: 2D array represnting an image
    :param p: percentage of compression
    :return: the compressed image
    """
    fft = FFT_CT2D(Image)
    fft_sorted = np.sort(np.abs(fft.reshape(-1)))
    thresh = fft_sorted[int(np.floor((1 - p) * len(fft_sorted)))]
    ind = np.abs(fft) > thresh
    return iFFT_CT2D(fft * ind).real
    

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
        "Description: Should print TRUE if the the result found by our FFT is the same as the one computed using our DFT and np.fft.fft, else it prints FALSE")
    print("The test is performed on a 1D 2^12 long array")
    print("Expected Output: Should return true for both methods")
    print("Output:")

    N = 2 ** 7
    signal = np.random.rand(N)
    fft = FFT_CT(signal)
    dft = DFT(signal)
    npfft = np.fft.fft(signal)
    print("Is FFT_CT equal to DFT ?")
    print(np.allclose(fft, dft))

    print("Is FFT_CT equal to numpy's FFT?")
    print(np.allclose(fft, npfft))

    print("__________________________")


def third():
    print("\nTHIRD TEST")
    print("__________________________")
    print(
        "Description: Should print TRUE if the the result found by our FFT is the same as the one computed using our DFT, else it prints FALSE")
    print("The test is performed on a 2D 2^10 long array")
    print("Output:")
    testVector_2D = np.random.rand(2 ** 7, 2 ** 7)
    print("Is FFT_CT2D equal to np.fft.fft2?")
    print(np.allclose(FFT_CT2D(testVector_2D), np.fft.fft2(testVector_2D)))
    print("Is FFT_CT2D equal to DFT2D?")
    print(np.allclose(FFT_CT2D(testVector_2D), DFT2D(testVector_2D)))
    print(np.allclose(iFFT_CT2D(FFT_CT2D(testVector_2D)), testVector_2D))
    print("__________________________")


def fourfth():
    print("\nFOURTH TEST")
    print("__________________________")
    print(
        "Description: Should print the time taken (in seconds) to compute the Discrete Fourier Transform using different algorithm ")
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

        start_time = timeit.default_timer()
        DFT(signal)
        #print("Time taken by DFT")
        #print(timeit.default_timer() - start_time)
        averageDFT = averageDFT + (timeit.default_timer() - start_time)

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


def fifth():
    print("\nFIFTH TEST")
    print("__________________________")
    print("Description: This test should determine which base case is the best for our recursion")
    print(
        "The test is performed on 1D array's ranging from 2^1 to 2^15, the average is made on testing each base case a thousand time."
        "From our graph, we expect the best base case to be between 2^0 and 2^9, so we will try base cases in this range")
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
            while i <= 1000:  # Iterates a thousand time over an array of the same base case and the length of array
                start_time = timeit.default_timer()
                FFT_CT_base(signal, 2 ** base)
                average = average + (timeit.default_timer() - start_time)
                i = i + 1
            average = float(average / (i - 1))
            totalaverage = totalaverage + average
            print("For a signal a signal of length 2^" + str(length) + " and a base case of 2^" + str(
                base) + " the average time to do the computation is " + str(average) + " seconds")
            length = length + 1
        averagetimes[base] = totalaverage
    print(averagetimes)
    
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
    plt.plot(N, CT_time)

    lists = sorted(bench_oldDFT_result.items())  # sorted by key, return a list of tuples
    N, oldDFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, oldDFT_time)
    plt.title("Ma bite sur ton front")
    plt.xlabel("Value of N (Size of the array)")
    plt.ylabel("Time taken (seconds)")
    plt.show()


def application():
    # https://e2eml.school/convert_rgb_to_grayscale.html
    image = np.asarray(Image.open('Koala.jpg'))  # Import the image

    grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to chekc its size and to make sure the compression is not impacted by the "imsave" method
    plt.imsave('GreyKoala.jpg', grayscale)
    size = os.path.getsize('GreyKoala.jpg') / 1000  # Get the size of the original image in kilobytes

    print(np.allclose(iFFT_CT2D(FFT_CT2D(grayscale)), grayscale))

    cmap = plt.get_cmap('gray')
    _, plots = plt.subplots(3, 2, figsize=(10, 8))
    plt.setp(plots, xticks=[], yticks=[])
    plots[0][0].set_title("Input image, size={}KB".format(size), size=10)
    plots[0][0].imshow(image, cmap, vmin=0, vmax=1)
    compression_percentage = [0.7, 0.2, 0.1, 0.05, 0.002]
    count1 = 1
    count2 = 0

    for j in compression_percentage:
        compressed = compress(grayscale, j)
        plt.imsave('compressed.jpg', compressed)  # Save the compressed image in the current package to get its size
        size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
        plots[count1][count2].set_title("{}% compressed, size={}KB".format(100*j, size), size=10)
        plots[count1][count2].imshow(compressed, cmap,  vmin=0, vmax=1)
        count1 += 1
        if count1 == 3:
            count1 = 0
            count2 += 1
    plt.show()

if __name__ == "__main__":
    #first()

    #second()

    #third()

    # fourfth()

    #fifth()

    #Create the graph
    # ourgraph()
    
    application()
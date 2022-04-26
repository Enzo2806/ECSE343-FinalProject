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

    Authors: Enzo Benoit-Jeannin and Thomas Dormart
    """

    y = np.zeros(inSignal.shape, dtype=complex)
    # Solution is based on the given formula in the assignment 4 instructions.
    N = inSignal.shape[0]  # N is the length of the input

    # Create a matrix with each cell storing the product of its indices
    # We need this matrix because in the formula f_hat = M @ f
    # both the index of the row and the column are needed to compute the value in M
    # But these two indices are multiplied, so we can create this matrix.
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

    Authors: Enzo Benoit-Jeannin and Thomas Dormart
    """

    N = inSignal.shape[0]  # N is the length of the input

    # The iDFT is the inverse discrete fourier transform
    # and it's just the DFT of the same signal, with a change of sign for s
    # multiplied by 1 / N
    return 1 / N * DFT(inSignal, 1)


def DFT2D(inSignal2D, s: int = -1):
    """
    Function to generate the 2-Dimensional discrete Fourier transform of the input signal.
    This 2D DFT is computed by first taking the DFT along the columns of the given signal,
    and then taking the DFT along the rows.
    :param inSignal2D: complex-valued (sampled) 2D DFT input array.
    :param s: sign parameter with default value -1 for the DFT vs. iDFT setting.
    :return: the generated DFT2D given the input signal.

    Authors: Enzo Benoit-Jeannin and Thomas Dormart
    """

    # Create arrays to store the results of performing DFT on columns and the final result
    result = np.zeros(inSignal2D.shape, dtype=complex)
    result_col = np.zeros(inSignal2D.shape, dtype=complex)

    i = 0  # counter variable
    # Iterate through all the columns in the given signal (by taking its transpose)
    # Compute the DFT along all these columns and store it in a temporary array result_col
    for col in inSignal2D.T:
        result_col[i] = DFT(col, s)
        i += 1
    result_col = result_col.T  # Transpose the array back to take compute the FFT along its rows

    i = 0  # counter variable

    # Iterate through all the rows of the temporary array
    # Compute the DFT along all these rows and store it in a the array of the final result
    for row in result_col:
        result[i] = DFT(row, s)
        i += 1

    return result


def iDFT2D(inSignal2D):
    """
    Function to generate the inverse 2-Dimensional discrete Fourier transform of the input signal.
    :param inSignal2D: complex-valued (sampled) 2D DFT input array.
    :return: the generated iDFT2D given the input signal.

    Authors: Enzo Benoit-Jeannin and Thomas Dormart
    """

    N = inSignal2D.shape[0]  # N is the length of the input
    # The iDFT2D is the 2D inverse discrete fourier transform
    # and it's just the 2D DFT of the same signal, with a change of sign for s
    # multiplied by 1 / N^2

    return (1 / N ** 2) * DFT2D(inSignal2D, 1)


def FFT_CT(inSignal, s: int = -1):
    """
    Function generating the FFT of the given signal using the Cooley-Tukey Algorithm.
    This algorithm runs faster than the previously implemnted DFT algorithm because it
    separates the given signal in two even and odd parts. It then recursively calls itself until the base case is met.
    :param inSignal: 1D (sampled) input signal numpy array with a power of 2 length
    :param s: sign parameter with default value -1 for the FFT vs. iFFT setting
    :return: returns the FFT of the input signal

    Authors: Enzo Benoit-Jeannin and Thomas Dormart
    """

    N = inSignal.shape[0]  # Get the length of the signal
    # Check if the length is not 0 and is a power of two.
    if N == 0:
        raise ValueError("Invalid signal: length 0")
    # Check for the base case of length 1 before checking for the power of 2.
    # If the length of the input signal is one, we simply return it.
    elif N == 1:
        return inSignal
    # If the length is not a power of two we throw an error.
    elif N % 2 != 0:
        raise ValueError("Invalid signal: length is not a power of 2")
    else:
        result = np.zeros(inSignal.shape, dtype=complex)
        # Separate the signal in two signals.
        # One containing the even indexed values and the other the odd indexed values.
        inSignal_e_hat = FFT_CT(inSignal[::2], s)
        inSignal_o_hat = FFT_CT(inSignal[1::2], s)

        # Here, we implemented our FFT differently from the instruction of the project.
        # Instead of forming the diag(e_a_hat) and diag(e_b_hat) diagnoal matrices, we generate a 1D vector containing
        # all the w values ranging from w^0 to w^n with w = exp(2j*pi / N). This allowed for better performance.
        w = np.exp(s * 2j * np.pi * np.arange(N) / N)

        # print("W is:")
        # print(w)
        #
        # print("w[:int(N / 2)] is")
        # print(w[:int(N / 2)])
        #
        # print("w[int(N / 2):] is")
        # print(w[int(N / 2):])
        #
        # w2 = np.exp(s * 2j * np.pi / N))
        # diag_e_a = np.full(int(N / 2), w2) ** np.arange(N / 2)
        #
        # print("diag_e_a is")
        # print(diag_e_a)
        # print("-diag_e_a is")
        # print(-diag_e_a)

        # Top N/2 values of the resulting numpy array is calculated using the equation given in the project report.
        result[0: int(N / 2)] = inSignal_e_hat + w[:int(N / 2)] * inSignal_o_hat
        # Bottom N/2 values of the resulting numpy array is calculated using the equation given in the project report.
        result[int(N / 2): N] = inSignal_e_hat + w[int(N / 2):] * inSignal_o_hat

        return result


def FFT_CT_Base(inSignal, base, s: int = -1):
    """
    Function generating the FFT of the given signal using the Cooley-Tukey Algorithm.
    This algorithm runs faster than the previously implemented DFT algorithm because it
    separates the given signal in two even and odd parts. It then recursively calls itself until the base case is met.

    This function is different from the FFT_CT() function because it accepts an extra argument specifying the
    base case value from which we compute the fourier transform of the signal using the DFT algorithm instead of
    splitting the signal in two again.

    :param inSignal: 1D (sampled) input signal numpy array with a power of 2 length
    :param base: Value of the base case form which we compute the fourier transform using the DFT algorithm
    :param s: sign parameter with default value -1 for the FFT vs. iFFT setting
    :return: returns the FFT of the input signal

    Authors: Enzo Benoit-Jeannin and Thomas Dormart
    """

    N = inSignal.shape[0]  # Get the length of the signal
    # Check if the length is not 0 and is a power of two.
    if N == 0:
        raise ValueError("Invalid signal: length 0")
    # Check for the base case of length specified in the argument of the function before checking for the power of 2.
    # If the length of the input signal is less than or equal to teh specified base value, we compute the DFT of the
    # signal and return it.
    elif N <= base:
        return DFT(inSignal, s)
    # If the length is not a power of two we throw an error.
    elif N % 2 != 0:
        raise ValueError("Invalid signal: length is not a power of 2")
    else:
        result = np.zeros(inSignal.shape, dtype=complex)
        # Separate the signal in two signals.
        # One containing the even indexed values and the other the odd indexed values.
        # We modified these two lines from the FFT_CT() function to call FFT() function instead.
        inSignal_e_hat = FFT_CT_Base(inSignal[::2], base, s)
        inSignal_o_hat = FFT_CT_Base(inSignal[1::2], base, s)

        # Here, we implemented our FFT differently from the instruction of the project.
        # Instead of forming the diag(e_a_hat) and diag(e_b_hat) diagnoal matrices, we generate a 1D vector containing
        # all the w values ranging from w^0 to w^n with w = exp(2j*pi / N). This allowed for better performance.
        w = np.exp(s * 2j * np.pi * np.arange(N) / N)

        # Top N/2 values of the resulting numpy array is calculated using the equation given in the project report.
        result[0: int(N / 2)] = inSignal_e_hat + w[:int(N / 2)] * inSignal_o_hat
        # Bottom N/2 values of the resulting numpy array is calculated using the equation given in the project report.
        result[int(N / 2): N] = inSignal_e_hat + w[int(N / 2):] * inSignal_o_hat

        return result


def FFT(inSignal, s: int = -1):
    """
    Function generating the FFT of the given signal using the Cooley-Tukey Algorithm.
    This algorithm runs faster than the previously implemented DFT algorithm because it
    separates the given signal in two even and odd parts. It then recursively calls itself until the base case is met.

    This function is different from the FFT_CT() function because it has a different base case that we determined
    in the tests available in the main method. This base case is 2^5=32, the function will call the dft algorithm
     for all signal of length less than or equal to 32 passed in this function.

    :param inSignal: 1D (sampled) input signal numpy array with a power of 2 length
    :param s: sign parameter with default value -1 for the FFT vs. iFFT setting
    :return: returns the FFT of the input signal

    Authors: Enzo Benoit-Jeannin and Thomas Dormart
    """
    return FFT_CT_Base(inSignal, 2 ** 5, s)


def iFFT_CT(inSignal):
    """
    Function generating the inverse FFT of the given signal using the Cooley-Tukey ALgorithm.
    This uses the FFT_CT function (with a base case of 1).
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
    Function generating the 2-Dimensional FFT of the given 2D signal using the Cooley-Tukey Algorithm we implemented.
    This uses the FFT_CT function (with a base case of 1).
    :param inSignal2D: 2D (sampled) input signal numpy array
    :param s: sign parameter with default value -1 for the FFT vs. iFFT setting
    :return: the 2D DFT of the input signal
    """
    # return FFT_CT(FFT_CT(np.transpose(inSignal2D)).T, s)
    result = np.zeros(inSignal2D.shape, dtype=complex)
    result_col = np.zeros(inSignal2D.shape, dtype=complex)

    i = 0
    for col in inSignal2D.T:
        result_col[i] = FFT_CT(col, s)
        i += 1
    i = 0
    for row in result_col.T:
        result[i] = FFT_CT(row, s)
        i += 1
    return result


def iFFT_CT2D(inSignal2D):
    """
    Function generating the inverse 2-Dimensional FFT of the given 2D signal using the Cooley-Tukey Algorithm we
    implemented. This uses the FFT_CT function (with a base case of 1).
    :param inSignal2D: complex-valued (sampled) 2D DFT input array.
    :return: the generated iDFT2D given the input signal.
    """
    N = inSignal2D.shape[0]  # N is the length of the input
    # The iFFT2D is the 2D inverse fast fourier transform
    # and it's just the 2D FFT of the same signal, with a change of sign for s
    # multiplied by 1 / N^2
    return 1 / N ** 2 * FFT_CT2D(inSignal2D, 1)


def iFFT(inSignal):
    """
    Function generating the inverse FFT of the given signal using the Cooley-Tukey ALgorithm.
    This uses the FFT function (with a base case different from 1).
    :param inSignal: complex-valued (sampled) 1D FFT input numpy array
    :return: the iFFT of the input signal
    """
    N = inSignal.shape[0]  # N is the length of the input
    # The iFFT is the inverse fast fourier transform
    # and it's just the FFT of the same signal, with a change of sign for s
    # multiplied by 1 / N
    return 1 / N * FFT(inSignal, 1)


def FFT2D(inSignal2D, s: int = -1):
    """
    Function generating the 2-Dimensional FFT of the given 2D signal using the Cooley-Tukey Algorithm we implemented.
    This uses the FFT function (with a base case different from 1).
    :param inSignal2D: 2D (sampled) input signal numpy array
    :param s: sign parameter with default value -1 for the FFT vs. iFFT setting
    :return: the 2D FFT of the input signal
    """

    # return FFT_CT(FFT_CT(np.transpose(inSignal2D)).T, s)
    result = np.zeros(inSignal2D.shape, dtype=complex)
    result_col = np.zeros(inSignal2D.shape, dtype=complex)

    i = 0
    for col in inSignal2D.T:
        result_col[i] = FFT(col, s)
        i += 1
    i = 0
    for row in result_col.T:
        result[i] = FFT(row, s)
        i += 1
    return result


def iFFT2D(inSignal2D):
    """
    Function generating the inverse 2-Dimensional FFT of the given 2D signal using the Cooley-Tukey Algorithm we
    implemented. This uses the FFT function (with a base case different from 1)
    :param inSignal2D: complex-valued (sampled) 2D FFT input array.
    :return: the generated iFFT2D given the input signal.
    """
    N = inSignal2D.shape[0]  # N is the length of the input
    # The iFFT2D is the 2D inverse fast fourier transform
    # and it's just the 2D FFT of the same signal, with a change of sign for s
    # multiplied by 1 / N^2
    return 1 / N ** 2 * FFT2D(inSignal2D, 1)


def compress(Img, p):
    """
    Function to compress an image. It first uses our FFT2D subroutine to get the fourier transform of the image.
    Then, we set to 0 the lowest fourier coefficients in the fourier domain of the image depending on the given
    argument p. We then use the inverse FFT2D subroutine we implemented to obtain the final compressed image.
    :param Img: 2D numpy array representing an image
    :param p: percentage of compression
    :return: the compressed image in the form of a 2D numpy array

    Example
    _______
    If p is 0.5, we will only keep 50% of the fourier coefficients by removing the smallest 50% other coefficients in
    the fourier domain.
    """

    fft = FFT2D(Img)  # Get the fft of the image

    # Sort the coefficients of the obtained fft (in absolute value) and reshape the 2D array to a 1D array
    fft_sorted = np.sort(np.abs(fft.reshape(-1)))

    # Get the index of the minimum value. We find the index of the element located at p*100% of the total length of the
    # array storing the fourier transform.
    index_min_value = int(np.floor((1 - p) * len(fft_sorted)))

    # Determine the minimum value of the coefficient we will keep in the fourier transform.
    min_value = fft_sorted[index_min_value]

    # Set to 0 all the values lower than the found minimum coefficient value.
    fft[np.abs(fft) < min_value] = 0
    # Apply the inverse FFT to get back to the new compressed image and return it.
    return iFFT2D(fft).real


def compress_DFT(Img, p):
    """
    THis function is the same as the "compress()" function but it uses the DFT subroutines for comparison.
    It therefore first uses our DFT2D subroutine to get the fourier transform of the image.
    Then, we set to 0 the lowest fourier coefficients in the fourier domain of the image depending on the given
    argument p. We then use the inverse DFT2D subroutine we implemented to obtain the final compressed image.
    :param Img: 2D numpy array representing an image
    :param p: percentage of compression
    :return: the compressed image in the form of a 2D numpy array

    Example
    _______
    If p is 0.5, we will only keep 50% of the fourier coefficients by removing the smallest 50% other coefficients in
    the fourier domain.
    """

    fft = DFT2D(Img)  # Get the fft of the image

    # Sort the coefficients of the obtained fft (in absolute value) and reshape the 2D array to a 1D array
    fft_sorted = np.sort(np.abs(fft.reshape(-1)))

    # Get the index of the minimum value. We find the index of the element located at p*100% of the total length of the
    # array storing the fourier transform.
    index_min_value = int(np.floor((1 - p) * len(fft_sorted)))

    # Determine the minimum value of the coefficient we will keep in the fourier transform.
    min_value = fft_sorted[index_min_value]

    # Set to 0 all the values lower than the found minimum coefficient value.
    fft[np.abs(fft) < min_value] = 0
    # Apply the inverse FFT to get back to the new compressed image and return it.
    return iDFT2D(fft).real


def first():
    print("Validation Tests")

    # Test the FFT_CT function
    print("\nFIRST TEST")
    print("__________________________")
    print("Description: Calls the FFT_CT function with a signal of length 0.")
    print("Expected Output: Should throw the following Exception: \"Invalid signal: length 0\" ")
    print("Output:")
    try:
        testVector = np.random.rand(0)  # Array of size 0
        FFT_CT(testVector)
    except Exception as e:  # Catch the error
        print(e)
    print("__________________________")


def second():
    # Test the FFT_CT function
    print("\nSECOND TEST")
    print("__________________________")
    print("Description: Calls the FFT_CT function with a signal of a length different from a power of 2.")
    print("Expected Output: Should throw the following Exception: \"Invalid signal: length is not a power of 2\" ")
    print("Output:")
    try:
        testVector = np.random.rand(3)  # Array of size different form a power of 2
        FFT_CT(testVector)
    except Exception as e:  # Catch the error
        print(e)
    print("__________________________")


def third(function=FFT_CT()):
    print("\nTHIRD TEST")
    print("__________________________")
    print("Description: Tests if the result found by our FFT is the same as the one computed using our ")
    print("DFT subroutine and numpy's own FFT algorithm.")
    print("The test is performed on multiple arrays. We save the result of the comparison for each array size in a ")
    print("separate array and print true if this array only contains true.")
    print("Expected Output: Should return true for both methods.")
    print("Output:")

    # Initialize numpy arrays to keep track of the results of the comparisons.
    comp_dft = np.array([], bool)
    comp_npfft = np.array([], bool)

    # For multiple arrays of different length (all power of 2)
    for N in (2 ** p for p in range(1, 13)):
        signal = np.random.rand(N)  # random signal
        fft = function(signal)    # Get fft
        dft = DFT(signal)       # Get dft
        npfft = np.fft.fft(signal)  # Get numpy fft
        np.append(comp_dft, np.allclose(fft, dft))  # Compare fft and numpy fft
        np.append(comp_npfft, np.allclose(fft, npfft))  # Compare fft and dft
    # Print results
    print("Is FFT_CT equal to DFT ?")
    print(comp_dft.all())
    print("Is FFT_CT equal to numpy's FFT?")
    print(comp_npfft.all())


def fourth(function=FFT_CT()):
    print("\nFOURTH TEST")
    print("__________________________")
    print("Description: This test benchmarks and compares our FFT_CT function with the naive DFT function.")
    print("The test is performed only once on multiple arrays of different length. We save the time taken per ")
    print("algorithm and per array length in a python dictionnary. We then generate a plot from it.")
    print("Expected Output: Should plot two curves for each algorithm: FFT and DFT. It will show the time taken")
    print("depending on the size of the array")
    print("with the best base case to use.")
    print("Output:")
    bench_CT_result = {}
    bench_oldDFT_result = {}

    for N in (2 ** p for p in range(0, 13)):
        signal = np.random.rand(N)
        start_time = timeit.default_timer()
        function(signal)
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


def fifth():
    print("\nFIFTH TEST")
    print("__________________________")
    print("Description: This test should determine which base case is the best for our recursion to compute the fft.")
    print("The test is performed on 1D array's ranging from 2^1 to 2^10, the average is made on testing each ")
    print("base case thirty times. From our graph in test 4, we expect the best base case to be between 2^0 and 2^9")
    print("so we will try base cases in this range. This test might take several minutes to finish even on HEDT")
    print("Expected Output: Should return the average time to compute FFT with each base case ")
    print("with the best base case to use.")
    print("Output:")

    results = np.zeros(10)

    # For all possible base cases (iterate through the possible power of 2)
    # We know thanks to the previous test that the fft algorithm is always faster than the dft for any signal of length
    # 2^9 or more so we iterate through base cases below 2^9.
    for base in range(9):
        # For each array of different length, we record the time taken to generate its fft 30 times and get an average
        # value for that particular array length and base case.
        # We add the average time to a variable that keeps track of the total time for this specific base case.
        # We then change array length and start again with the same base case and add it to the same variable.
        # We do this for all base cases and obtain an array storing accumulated average times for each base case.

        totalAverage_base = 0  # keeps track of how each base case performs.

        for length in range(10):  # power of 2 used for the length of the signal
            signal = np.random.rand(2 ** length)    # generate random signal of length of power of 2
            i = 0   # counting variable
            average = 0     # keeps track of the average time taken to compute the fft given an array and a base case

            # Iterates thirty times to to get an average value of the time to
            # compute the fft of a same array with a same base case.
            while i < 30:
                start_time = timeit.default_timer()     # Start timer
                FFT_CT_Base(signal, 2 ** base)          # Compute fft of the current array using the current base case
                average = average + (timeit.default_timer() - start_time)   # End timer
                i = i + 1

            average = float(average / (i - 1))      # Take the average of all the time recorder in the previous loop

            totalAverage_base = totalAverage_base + average     # Add this average time to

        results[base] = totalAverage_base

    print(results)
    print("The base case with the best performance on average is:")



def fourth():
    print("\nFOURTH TEST")
    print("__________________________")
    print("Description: Should print TRUE if the the result found by our FFT is the same as the one computed using our "
        "DFT, else it prints FALSE")
    print("The test is performed on a 2D 2^12 long array")
    print("Output:")
    testVector_2D = np.random.rand(2 ** 12, 2 ** 12)
    fft = FFT_CT2D(testVector_2D)
    npfft = np.fft.fft2(testVector_2D)
    dft = DFT2D(testVector_2D)
    # fftbase= FFT_CT2D_base(testVector_2D,-1)
    print("Is FFT_CT2D equal to numpy's 2D FFT?")
    print(np.allclose(fft, npfft))

    print("Is FFT_CT2D equal to DFT2D?")
    print(np.allclose(fft, dft))

    # print("Is FFT_CT2D_base equal to numpy's 2D FFT?")
    # print(np.allclose(fftbase, npfft))

    # print("Is FFT-CT2D_base equal to FFT_2D?")
    # print(np.allclose(fftbase,fft))
    # print(np.allclose(iFFT_CT2D(FFT_CT2D(testVector_2D)), testVector_2D))

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
    ifftbase = iFFT(signal)
    print("Is iFFT_CT equal to iDFT ?")
    print(np.allclose(ifft, idft))

    print("Is iFFT_CT equal to numpy's iFFT?")
    print(np.allclose(ifft, inpfft))

    print("Is iFFT_CT_base equal to iFFT_CT?")
    print(np.allclose(ifftbase, ifft))

    print("Is iFFT_CT_base equal to iDFT?")
    print(np.allclose(ifftbase, idft))
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
        # print("Time taken by FFT_CT:")
        # print(timeit.default_timer() - start_time)
        averagefftct = averagefftct + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        np.fft.fft(signal)
        # print("Time taken by np.fft.fft:")
        # print(timeit.default_timer() - start_time)
        averagefftnp = averagefftnp + (timeit.default_timer() - start_time)

        # start_time = timeit.default_timer()
        # DFT(signal)
        # print("Time taken by DFT")
        # print(timeit.default_timer() - start_time)
        # averageDFT = averageDFT + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        FFT(signal, 2 ** 5)
        # print("Time taken by FFT_CT_BASE:")
        # print(timeit.default_timer() - start_time)
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
        signal = np.random.rand(2 ** 8, 2 ** 8)

        start_time = timeit.default_timer()
        FFT_CT2D(signal)
        averagefftct = averagefftct + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        np.fft.fft2(signal)
        averagefftnp = averagefftnp + (timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        DFT2D(signal)
        averageDFT = averageDFT + (timeit.default_timer() - start_time)

        # start_time = timeit.default_timer()
        # FFT_CT2D_base(signal, 2 ** 5)
        # averagefftbase = averagefftbase + (timeit.default_timer() - start_time)

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
        iFFT(signal)
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
        "Description: Should print the average time taken (in seconds) to compute the inverse 2D Discrete Fourier Transform using different algorithm ")
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


def ourgraph2D():
    bench_2DFFT_result = {}
    bench_2DDFT_result = {}

    for N in (2 ** p for p in range(0, 10)):
        signal = np.random.rand(N, N)

        start_time = timeit.default_timer()
        fft = FFT2D(signal)
        bench_2DFFT_result[N] = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        dft = DFT2D(signal)
        bench_2DDFT_result[N] = timeit.default_timer() - start_time
        print(np.allclose(fft, np.fft.fft2(signal)))

    lists = sorted(bench_2DFFT_result.items())  # sorted by key, return a list of tuples
    N, FFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, FFT_time, label="FFT")

    lists = sorted(bench_2DDFT_result.items())  # sorted by key, return a list of tuples
    N, DFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, DFT_time, label="DFT")

    plt.title("Graph comparing the time efficiency of the DFT2D and FFT2D algorithm")
    plt.xlabel("Value of N (Size of the array)")
    plt.ylabel("Time taken (seconds)")
    plt.legend()
    plt.show()


def ourgraphi2D():
    bench_i2DFFT_result = {}
    bench_i2DDFT_result = {}

    for N in (2 ** p for p in range(0, 10)):
        signal = np.random.rand(N, N)
        start_time = timeit.default_timer()
        fft = iFFT2D(signal)
        bench_i2DFFT_result[N] = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        dft = iDFT2D(signal)
        bench_i2DDFT_result[N] = timeit.default_timer() - start_time
        print(np.allclose(fft, np.fft.ifft2(signal)))

    lists = sorted(bench_i2DFFT_result.items())  # sorted by key, return a list of tuples
    N, FFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, FFT_time, label="iFFT")

    lists = sorted(bench_i2DDFT_result.items())  # sorted by key, return a list of tuples
    N, DFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, DFT_time, label="iDFT")

    plt.title("Graph comparing the time efficiency of the iDFT2D and iFFT2D algorithm")
    plt.xlabel("Value of N (Size of the array)")
    plt.ylabel("Time taken (seconds)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    first()

    # second()

    third2()

    # fourth()

    # fifth()

    # sixth()

    # seventh()

    # eighth()

    # ninth()

    # tenth()

    # Create the graph
    # ourgraph()
    # ourgraph2D()
    # ourgraphi2D()

    # Application
    print("The application of the FFT algorithm we chose consists of compressing a given image to reduce its size.")
    print("This is done by taking the fast fourier transform of an input image (in grayscale) and by removing "
          "(set to 0) the lowest frequencies of that fft depending on a threshold we specify. We then use the inverse "
          "fft algorithm we wrote to go generate the compressed image.")
    print("We defined a function named \"compress\" that takes as argument an image and the percentage of frequencies "
          "to keep in the fast fourier transform. For instance, compress(imageTest, 0.7) will remove 30% of all the "
          "frequencies in the fourier domain (only the lowest ones).")

    print("\nFIRST APPLICATION TEST")
    print("__________________________")
    print("Description: This first application test will compress a koala image multiple times.")
    print("Expected Output: The output should be a matplotlib plot, showing the input image and all 5 resulting "
          "compressed images with their respective new sizes.")
    print("Output:")

    # https://e2eml.school/convert_rgb_to_grayscale.html
    image = np.asarray(Image.open('Koala.jpg'))  # Import the image
    grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    plt.imsave('GreyKoala.jpg', grayscale)
    size = os.path.getsize('GreyKoala.jpg') / 1000  # Get the size of the original image in kilobytes

    cmap = plt.get_cmap('gray')
    _, plots = plt.subplots(3, 2, figsize=(10, 8))
    plt.setp(plots, xticks=[], yticks=[])
    plots[0][0].set_title("Input image, size={}KB".format(size), size=10)
    plots[0][0].imshow(grayscale, cmap, vmin=0, vmax=1)
    compression_percentage = [0.7, 0.2, 0.1, 0.05, 0.002]

    count1 = 1
    count2 = 0

    for j in compression_percentage:
        compressed = compress(grayscale, j)

        plt.imsave('compressed.jpg', compressed)  # Save the compressed image in the current package to get its size
        size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
        plots[count1][count2].set_title(
            "Compression removed {}% of input image\nNew size={}KB".format(100 - 100 * j, size), size=10)
        plots[count1][count2].imshow(compressed, cmap, vmin=0, vmax=1)
        count1 += 1
        if count1 == 3:
            count1 = 0
            count2 += 1
    plt.show()

    # print("\nSECOND APPLICATION TEST")
    # print("__________________________")
    # print("Description: This second application tests the compress_DFT() method"
    #       "of the same koala image.")
    # print("Expected Output:")
    # print("Output: ")
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
    #     compressed = compress_DFT(grayscale, j)
    #     plt.imsave('compressed.jpg', compressed)  # Save the compressed image in the current package to get its size
    #     size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
    #     plots[count1][count2].set_title(
    #         "Compression removed {}% of input image\nNew size={}KB".format(100 - 100 * j, size), size=10)
    #     plots[count1][count2].imshow(compressed, cmap, vmin=0, vmax=1)
    #     count1 += 1
    #     if count1 == 3:
    #         count1 = 0
    #         count2 += 1
    # plt.show()
    #
    # print("\nTHIRD APPLICATION TEST")
    # print("__________________________")
    # print("Description: This third application test will compare the dft and fft algorithms in the compression"
    #       "of the same koala image.")
    # print("Expected Output: Benchmark of 4 compressions of 4 different images comparing the DFT and FFT algorithms.")
    # print("Output: ")

    # Import all four images following the same procedure as in the firts test.
    # Convert RGB Pictures to greyscale, save them to check their size (which changed since we converted them to
    # greyscale).
    # image_VieuxLyon = np.asarray(Image.open('VieuxLyon.jpg'))  # Import the image
    # grayscale_VieuxLyon = np.dot(image_VieuxLyon[..., :3],
    #                              [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    # plt.imsave('VieuxLyon.jpg', grayscale_VieuxLyon, cmap='gray')
    # size_VieuxLyon = os.path.getsize('VieuxLyon.jpg') / 1000  # Get the size of the original image in kilobytes

    # image_MoulinRouge = np.asarray(Image.open('MoulinRouge.jpg'))  # Import the image
    # grayscale_MoulinRouge = np.dot(image_MoulinRouge[..., :3],
    #                                [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    # plt.imsave('MoulinRouge.jpg', grayscale_MoulinRouge, cmap='gray')
    # size_MoulinRouge = os.path.getsize('MoulinRouge.jpg') / 1000  # Get the size of the original image in kilobytes
    #
    # image_Koala = np.asarray(Image.open('Koala.jpg'))  # Import the image
    # grayscale_Koala = np.dot(image_Koala[..., :3],
    #                          [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    # plt.imsave('Koala.jpg', grayscale_Koala, cmap='gray')
    # size_Koala = os.path.getsize('Koala.jpg') / 1000  # Get the size of the original image in kilobytes
    #
    # image_Fourviere = np.asarray(Image.open('Fourviere.jpg'))  # Import the image
    # grayscale_Fourviere = np.dot(image_Fourviere[..., :3],
    #                              [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    # plt.imsave('Fourviere.jpg', grayscale_Fourviere, cmap='gray')
    # size_Fourviere = os.path.getsize('Fourviere.jpg') / 1000  # Get the size of the original image in kilobytes
    #
    # # Configure the plot by adding the input images in a first column.
    # cmap = plt.get_cmap('gray')
    # _, plots = plt.subplots(4, 2, figsize=(10, 8))
    # plt.setp(plots, xticks=[], yticks=[])
    # # plots[0][0].set_title("Input 4096x4096 image, size={}KB".format(size_VieuxLyon), size=10)
    # # plots[0][0].imshow(grayscale_VieuxLyon, cmap, vmin=0, vmax=1)
    # plots[1][0].set_title("Input 2048x2048 image, size={}KB".format(size_MoulinRouge), size=10)
    # plots[1][0].imshow(grayscale_MoulinRouge, cmap, vmin=0, vmax=1)
    # plots[2][0].set_title("Input 1024x1024 image, size={}KB".format(size_Koala), size=10)
    # plots[2][0].imshow(grayscale_Koala, cmap, vmin=0, vmax=1)
    # plots[3][0].set_title("Input 512x512 image, size={}KB".format(size_Fourviere), size=10)
    # plots[3][0].imshow(grayscale_Fourviere, cmap, vmin=0, vmax=1)
    #
    # # Variables to keep track of the time taken for each compression for both DFT and FFT algorithm
    # bench_FFT_result = {}
    # bench_DFT_result = {}

    # # 4096x4096 picture compression using DFT
    # start_time = timeit.default_timer()
    # compressed = compress_DFT(grayscale_VieuxLyon, 0.009)
    # bench_DFT_result[4096] = timeit.default_timer() - start_time

    # # 4096x4096 picture compression using FFT
    # start_time = timeit.default_timer()
    # compressed = compress(grayscale_VieuxLyon, 0.009)
    # bench_FFT_result[4096] = timeit.default_timer() - start_time
    #
    # # Save the compressed image in the current package to get its size
    # plt.imsave('compressed.jpg', compressed, cmap='gray')
    # plots[0][1].set_title("Compression removed 99,1% of input image", size=10)
    # plots[0][1].imshow(compressed, cmap, vmin=0, vmax=1)

    # 2048x2048 picture compression using FFT
    # start_time = timeit.default_timer()
    # compressed = compress(grayscale_MoulinRouge, 0.009)
    # bench_FFT_result[2048] = timeit.default_timer() - start_time
    # # Save the compressed image in the current package to get its size
    # plt.imsave('compressed.jpg', compressed, cmap='gray')
    # size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
    # plots[1][1].imshow(compressed, cmap, vmin=0, vmax=1)
    #
    # # 2048x2048 picture compression using DFT
    # start_time = timeit.default_timer()
    # compressed = compress_DFT(grayscale_MoulinRouge, 0.009)
    # bench_DFT_result[2048] = timeit.default_timer() - start_time
    #
    # # 1024x1024 picture compression using FFT
    # start_time = timeit.default_timer()
    # compressed = compress(grayscale_Koala, 0.009)
    # bench_FFT_result[1024] = timeit.default_timer() - start_time
    # # Save the compressed image in the current package to get its size
    # plt.imsave('compressed.jpg', compressed, cmap='gray')
    # size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
    # plots[2][1].imshow(compressed, cmap, vmin=0, vmax=1)
    #
    # # 1024x1024 picture compression using DFT
    # start_time = timeit.default_timer()
    # compressed = compress_DFT(grayscale_Koala, 0.009)
    # bench_DFT_result[1024] = timeit.default_timer() - start_time
    #
    # # 512x512 picture compression using FFT
    # start_time = timeit.default_timer()
    # compressed = compress(grayscale_Fourviere, 0.009)
    # bench_FFT_result[512] = timeit.default_timer() - start_time
    # # Save the compressed image in the current package to get its size
    # plt.imsave('compressed.jpg', compressed, cmap='gray')
    # size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes
    # plots[3][1].imshow(compressed, cmap, vmin=0, vmax=1)
    #
    # # 1024x1024 picture compression using DFT
    # start_time = timeit.default_timer()
    # compressed = compress_DFT(grayscale_Fourviere, 0.009)
    # bench_DFT_result[512] = timeit.default_timer() - start_time
    # plt.show()
    #
    # print("FFT")
    # print(bench_FFT_result)
    # print("DFT")
    # print(bench_DFT_result)
    #
    # lists = sorted(bench_FFT_result.items())  # sorted by key, return a list of tuples
    # N, FFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    # plt.plot(N, FFT_time, label="FFT")
    #
    # lists = sorted(bench_DFT_result.items())  # sorted by key, return a list of tuples
    # N, oldDFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    # plt.plot(N, oldDFT_time, label="DFT")
    #
    # plt.title("Graph comparing the time efficiency of the DFT and FFT algorithm for the compression application.")
    # plt.xlabel("Value of N (Size of the image)")
    # plt.ylabel("Time taken (seconds)")
    # plt.legend()
    # plt.show()

    # # 3D Tests
    # N = 2 ** 8
    # signal = np.random.rand(N, N, 3)
    # print(signal)
    # fft = FFT3D(signal)
    # print(np.allclose(fft, np.fft.fftn(signal)))

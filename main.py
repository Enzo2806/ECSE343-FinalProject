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

        # Here, we implemented our FFT using the first equation given in the  instruction of the project.
        # Instead of forming the diag(e_a_hat) and diag(e_b_hat) diagonal matrices and use the matrix form of the
        # equation given in the instructions, we generate a 1D vector containing all the w values ranging from w^0 to
        # w^n with w = exp(2j*pi / N). The recursion remains the same, but instead of
        w = np.exp(s * 2j * np.pi * np.arange(N) / N)

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

    dft = DFT2D(Img)  # Get the dft of the image

    # Sort the coefficients of the obtained fft (in absolute value) and reshape the 2D array to a 1D array
    dft_sorted = np.sort(np.abs(dft.reshape(-1)))

    # Get the index of the minimum value. We find the index of the element located at p*100% of the total length of the
    # array storing the fourier transform.
    index_min_value = int(np.floor((1 - p) * len(dft_sorted)))

    # Determine the minimum value of the coefficient we will keep in the fourier transform.
    min_value = dft_sorted[index_min_value]

    # Set to 0 all the values lower than the found minimum coefficient value.
    dft[np.abs(dft) < min_value] = 0
    # Apply the inverse DFT to get back to the new compressed image and return it.
    return iDFT2D(dft).real


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


def third(function1=FFT_CT, function2=DFT, function3=np.fft.fft):
    # This test can be reused to compare any 3 functions returning an a DFT or an iDFT.
    name1 = function1.__name__
    name2 = function2.__name__
    name3 = function3.__name__
    if name1 == "FFT_CT":
        print("\nTHIRD TEST")
    print("__________________________")
    print("Description: Tests if the result found by our " + name1 + " is the same as the one computed using our ")
    print(name2 + " subroutine and numpy's " + name3 + " algorithm.")
    print("The test is performed on multiple arrays. We save the result of the comparison for each array size in a ")
    print("separate array and print true if this array only contains true.")
    print("Expected Output: Should return True for both methods.")
    print("Output:")

    # Initialize numpy arrays to keep track of the results of the comparisons.
    comp_dft = np.array([], bool)
    comp_npfft = np.array([], bool)

    # For multiple arrays of different length (all power of 2)
    for N in (2 ** p for p in range(0, 9)):
        if name1 == "FFT" or name1 == "FFT_CT" or name1 == "iFFT" or name1 == "iFFT_CT":
            signal = np.random.rand(N)  # random signal of 1 dimension
        else:
            signal = np.random.rand(N, N)  # random signal of 2 dimension

        fft = function1(signal)  # Get fft
        dft = function2(signal)  # Get dft
        npfft = function3(signal)  # Get numpy fft
        comp_dft = np.append(comp_dft, np.allclose(fft, dft))  # Compare fft and numpy fft
        comp_npfft = np.append(comp_npfft, np.allclose(fft, npfft))  # Compare fft and dft
    # Print results
    print("Is " + name1 + " equal to " + name2 + " ?")
    print(comp_dft.all())
    print("Is " + name1 + " equal to numpy's " + name3 + "?")
    print(comp_npfft.all())


def fourth():
    print("\nFOURTH TEST")
    print("__________________________")
    print("Description: This test benchmarks and compares our FFT_CT function with the naive DFT function.")
    print("The test computes a same FFT/DFT 30 times on each array and saves their corresponding average times in ")
    print("a python dictionary. We vary the length of the array we apply our algorithms to in order to show the FFT")
    print("is more efficient on larger arrays compare to the DFT algorithm. We then generate a plot from it.")
    print("Expected Output: Should plot two curves for each algorithm: FFT/DFT. It will show the average time taken")
    print("depending on the size of the array.")
    print("Output:")

    # Initialize dictionaries to keep track of which array length made the fft/dft took which time
    bench_FFT_CT_result = {}
    bench_DFT_result = {}

    # Iterate through different array length, all of length of power of 2
    for N in (2 ** p for p in range(0, 13)):
        signal = np.random.rand(N)  # generate random signal
        average_FFT = 0
        average_DFT = 0
        for j in range(30):
            start_time = timeit.default_timer()  # Start timer
            FFT_CT(signal)  # compute fft ( call the FFT_CT function, later tests will call a different function)
            average_FFT += timeit.default_timer() - start_time  # Stop timer

            start_time = timeit.default_timer()  # Start timer
            DFT(signal)  # compute dft
            average_DFT += timeit.default_timer() - start_time  # Stop timer
        j = 30
        bench_FFT_CT_result[N] = average_FFT / j
        bench_DFT_result[N] = average_DFT / j

    lists = sorted(bench_FFT_CT_result.items())  # sorted by key, return a list of tuples
    N, FFT_CT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, FFT_CT_time, label="FFT")  # Plot the curve for the FFT_CT and add a label

    lists = sorted(bench_DFT_result.items())  # sorted by key, return a list of tuples
    N, DFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, DFT_time, label="DFT")  # Plot the curve for the DFT and add a label

    plt.title("Graph comparing the time efficiency of the DFT and FFT functions")
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
    print("so we will try base cases in this range. This test might take several minutes to finish.")
    print("Expected Output: Should return the average time to compute an FFT of arrays of different length ")
    print("corresponding to each base case. It should also return the best base case to use.")
    print("Output:")

    results = np.zeros(10)  # Initialize array to store results in

    # For all possible base cases (iterate through the possible power of 2)
    # We know thanks to the previous test that the fft algorithm is always faster than the dft for any signal of length
    # 2^9 or more so we iterate through base cases below 2^9.
    for base in range(10):
        # For each array of different length, we record the time taken to generate its fft 30 times and get an average
        # value for that particular array length and base case.
        # We add the average time to a variable that keeps track of the total time for this specific base case.
        # We then change array length and start again with the same base case and add it to the same variable.
        # We do this for all base cases and obtain an array storing accumulated average times for each base case.

        totalAverage_base = 0  # keeps track of how each base case performs.

        for length in range(10):  # power of 2 used for the length of the signal
            signal = np.random.rand(2 ** length)  # generate random signal of length of power of 2
            i = 0  # counting variable
            average = 0  # keeps track of the average time taken to compute the fft given an array and a base case

            # Iterates thirty times to to get an average value of the time to
            # compute the fft of a same array with a same base case.
            while i < 30:
                start_time = timeit.default_timer()  # Start timer
                FFT_CT_Base(signal, 2 ** base)  # Compute fft of the current array using the current base case
                average = average + (timeit.default_timer() - start_time)  # End timer
                i = i + 1

            average = float(average / (i - 1))  # Take the average of all the time recorder in the previous loop

            totalAverage_base = totalAverage_base + average  # Add this average time to

        results[base] = totalAverage_base

    best = np.argmin(results)  # Find index of minimum total average time to find best base case
    steps = 2 ** np.arange(10)  # Array to keep track steps for the table.

    # Make matrix with left columns containing base case and right column containing its corresponding average
    # time for the table.
    table_data = np.column_stack((steps, results))

    column_labels = ["Base Case Value", "Average time"]  # Name of columns for table
    colors = plt.cm.BuPu(np.full(len(column_labels), 0.1))  # Color of column titles for table
    highlight_color = '#E8E190'  # Yellow color to highlight result

    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=table_data, colLabels=column_labels, colColours=colors, loc="center")  # Design table

    best_cell1 = table[best + 1, 0]  # get the cells corresponding to the base case (do + 1 because of column headers)
    best_cell2 = table[best + 1, 1]
    best_cell1.set_facecolor(highlight_color)  # Highlight the rows
    best_cell2.set_facecolor(highlight_color)

    plt.title("Table showing average time taken for each base case when \napplied to arrays of different length. "
              "The best base case is: " + str(2 ** best))
    plt.show()
    print("The base case with the best performance on average is: " + str(2 ** best))
    print(
        "The rest of the program will now use the function called FFT() which uses this base case of " + str(2 ** best))


def sixth(function1=FFT_CT, function2=DFT, function3=FFT):
    # This test can be reused to compare any 3 functions returning an a fourier transform or an inverse
    # fourier transform.
    name1 = function1.__name__
    name2 = function2.__name__
    name3 = function3.__name__
    if name1 == "FFT_CT":
        print("\nSIXTH TEST")
    print("__________________________")
    print("Description: This test benchmarks/compares our "+name3+" function with the "+name2+" function and "+name1)
    print("function. The " + name3 + " function has base case 2^32 while " + name1 + " has a base case of 1.")
    print(
        "The test computes a same "+name1+"/"+name2+"/"+name3+" 30 times on each array and saves their corresponding")
    print("average times in a dictionary. We vary the length of the array we apply the algorithms to in order to")
    print(
        "show the " + name3 + " is more efficient on larger arrays compare to the "+name2+"/"+name1 + " algorithms.")
    print("We generate a plot from it.")
    print("Expected Output: Should plot three curves for each algorithm: " + name1 + ", " + name3 + " and " + name2)
    print("It will show the time taken depending on the size of array.")
    print("Output:")

    # Initialize dictionaries to keep track of which array length made the fft/dft took which time
    bench_FFT_CT_result = {}
    bench_DFT_result = {}
    bench_FFT_result = {}

    # Iterate through different array length, all of length of power of 2
    for N in (2 ** p for p in range(0, 9)):

        if name1 == "FFT" or name1 == "FFT_CT" or name1 == "iFFT" or name1 == "iFFT_CT":
            signal = np.random.rand(N)  # generate random signal of 1 dimension
        else:
            signal = np.random.rand(N, N)  # generate random signal of 2 dimension
        average_FFT_CT = 0
        average_DFT = 0
        average_FFT = 0
        for j in range(30):
            start_time = timeit.default_timer()  # Start timer
            function1(signal)  # compute fft using FFT_CT function (base case of 1)
            average_FFT_CT += timeit.default_timer() - start_time  # Stop timer

            start_time = timeit.default_timer()  # Start timer
            function2(signal)  # compute dft
            average_DFT += timeit.default_timer() - start_time  # Stop timer

            start_time = timeit.default_timer()  # Start timer
            function3(signal)  # compute fft using FFT function (base case of 2^32)
            average_FFT += timeit.default_timer() - start_time  # Stop timer

        j = 30
        bench_FFT_CT_result[N] = average_FFT_CT / j
        bench_DFT_result[N] = average_DFT / j
        bench_FFT_result[N] = average_FFT / j

    lists = sorted(bench_FFT_CT_result.items())  # sorted by key, return a list of tuples
    N, FFT_CT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, FFT_CT_time, label=name1)  # Plot the curve for the FFT_CT and add a label

    lists = sorted(bench_DFT_result.items())  # sorted by key, return a list of tuples
    N, DFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, DFT_time, label=name2)  # Plot the curve for the DFT and add a label

    lists = sorted(bench_FFT_result.items())  # sorted by key, return a list of tuples
    N, FFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, FFT_time, label=name3)  # Plot the curve for the FFT and add a label

    plt.title("Graph comparing the time efficiency of the " + name3 + ", " + name2 + "\n and " + name1 + " functions")
    plt.xlabel("Value of N (Size of the array)")
    plt.ylabel("Time taken (seconds)")
    plt.legend()
    plt.show()


def seventh():
    # This test is the same as the third one, except that we check that our FFT function is correct, not the FFT_CT one
    print("\nSEVENTH TEST")
    third(FFT)


def eighth():
    # This test is the same as the third one, except that we check that our iFFT function is correct, not the FFT_CT one
    print("\nEIGHTH TEST")
    third(iFFT, iDFT, np.fft.ifft)


def ninth():
    # This test is the same as the sixth one, except that we plot our iFFT, iDFT and iFFT_CT functions
    print("\nNINTH TEST")
    sixth(iFFT_CT, iDFT, iFFT)


def tenth(function1=FFT, function2=iFFT):
    name1 = function1.__name__
    name2 = function2.__name__
    if name1 == "FFT":
        print("\nTENTH TEST")
    print("__________________________")
    print("Description: Test if the the result found by applying our " + name2 + " on a signal resulting from")
    print("the " + name1 + " of a signal is equal to the original signal. We compare the results on different arrays")
    print("of different sizes.")
    print("Expected Output: Should print True")
    print("Output:")
    # Initialize numpy array to keep track of the results of the comparisons.
    comparisons = np.array([], bool)

    # For multiple arrays of different length (all power of 2)
    for N in (2 ** p for p in range(1, 13)):
        if name1 == "FFT" or name1 == "FFT_CT" or name1 == "iFFT" or name1 == "iFFT_CT":
            signal = np.random.rand(N)  # random signal of 1 dimension
        else:
            signal = np.random.rand(N, N)  # random signal of 2 dimension
        # Compare fft and numpy fft
        comparisons = np.append(comparisons, np.allclose(signal, function2(function1(signal))))
    # Print results
    print("Is " + name2 + "(" + name1 + "(signal)) equal to the original signal ?")
    print(comparisons.all())


def eleventh():
    # This test is the same as the tenth one, except that we check the correctness of our iFFT2 and FFT2D functions
    print("\nELEVENTH TEST")
    tenth(FFT2D, iFFT2D)


def twelfth():
    # This test is the same as the third one, except that we check that our FFT2D function is correct, not the FFT_CT
    print("\nTWELFTH TEST")
    third(FFT2D, DFT2D, np.fft.fft2)


def thirteenth():
    # This test is the same as the sixth one, except that we plot our FFT2D, DFT2D and FFT_CT2D functions
    print("\nTHIRTEENTH TEST")
    sixth(FFT_CT2D, DFT2D, FFT2D)


def fourteenth():
    # This test is the same as the third one, except that we check that our iFFT2D function is correct, not the FFT_CT
    print("\nFOURTEENTH TEST")
    third(iFFT2D, iDFT2D, np.fft.ifft2)


def fifteenth():
    # This test is the same as the sixth one, except that we plot our iFFT2D, iDFT2D and iFFT_CT2D functions
    print("\nFIFTEENTH TEST")
    sixth(iFFT_CT2D, iDFT2D, iFFT2D)


def app_first():
    cmap = plt.get_cmap('gray')  # To save grayscale pictures / plot grayscale images
    print("\nFIRST APPLICATION TEST")
    print("__________________________")
    print("Description: This first application test will compress a simple square image of 32 pixels width.")
    print("Expected output: The output should be a matplotlib plot, showing the input image and the resulting "
          "compressed image with the new size of that image.")
    print("Output:")
    # https://e2eml.school/convert_rgb_to_grayscale.html
    image = np.asarray(Image.open('Fourviere.jpg'))  # Import the image
    grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression comparison
    # is not impacted by the "imsave()" method.
    plt.imsave('Fourviere.jpg', grayscale, cmap=cmap)
    size = os.path.getsize('Fourviere.jpg') / 1000  # Get the size of the original image in kilobytes

    # Initialize the plot
    _, plots = plt.subplots(2, 1, figsize=(10, 8))
    plt.setp(plots, xticks=[], yticks=[])
    plots[0].set_title("Input image, size={}KB".format(size), size=10)  # Plot the original image, display its size
    plots[0].imshow(grayscale, cmap, vmin=0, vmax=1)

    # Get the compressed image, remove 96% of fourier coefficients in the fourier domain
    compressed = compress(grayscale, 0.04)

    # Save the compressed image in the current package to get its size
    plt.imsave('compressed.jpg', compressed, cmap=cmap)
    size = os.path.getsize('compressed.jpg') / 1000  # Get the size of the compressed image in kilobytes

    plots[1].set_title("Compression removed 96% of fourier coefficients in the image"
                       "\nNew size={}KB".format(size), size=10)
    plots[1].imshow(compressed, cmap, vmin=0, vmax=1)

    plt.show()


def app_second(function=compress):
    cmap = plt.get_cmap('gray')  # To save grayscale pictures / plot grayscale images
    func_name = function.__name__
    if func_name == "compress":
        print("\nSECOND APPLICATION TEST")
    print("__________________________")
    print("Description: This application test will compress a koala image multiple times using the " + func_name)
    print("function. This tests if this function compresses the image ase expected.")
    print("Expected Output: The output should be a matplotlib plot, showing the input image and all 5 resulting "
          "compressed images with their respective new sizes.")
    print("Output:")

    # https://e2eml.school/convert_rgb_to_grayscale.html
    image = np.asarray(Image.open('Koala.jpg'))  # Import the image
    grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression comparison
    # is not impacted by the "imsave()" method.
    plt.imsave('Koala.jpg', grayscale, cmap=cmap)
    size = os.path.getsize('Koala.jpg') / 1000  # Get the size of the original image in kilobytes

    # Initialize the plot
    _, plots = plt.subplots(3, 2, figsize=(10, 8))
    plt.setp(plots, xticks=[], yticks=[])
    plots[0][0].set_title("Input image, size={}KB".format(size), size=10)  # Plot the original image, display its size
    plots[0][0].imshow(grayscale, cmap, vmin=0, vmax=1)

    compression_percentage = [0.05, 0.03, 0.02, 0.01, 0.002]  # Different percentage values to compress the image with.

    # Two variables to keep track of the next slot on the plot
    count1 = 1
    count2 = 0

    # For all coefficients in the percentage array, compress the image using that percentage and add it to the plot.
    for percent in compression_percentage:
        compressed = function(grayscale, percent)  # Compress the image

        # Save the compressed image in the a folder in the current package to get its size.
        plt.imsave("Koala_compressed/compressed_{}.jpg".format(percent), compressed, cmap=cmap)
        # Get the size of the compressed image in kilobytes.
        size = os.path.getsize("Koala_compressed/compressed_{}.jpg".format(percent)) / 1000
        plots[count1][count2].set_title("Compression removed {}% of fourier coefficients\nin the image"
                                        "New size={}KB".format(100 - 100 * percent, size), size=10)
        plots[count1][count2].imshow(compressed, cmap, vmin=0, vmax=1)

        count1 += 1
        if count1 == 3:
            count1 = 0
            count2 += 1
    plt.show()


def app_third():
    print("\nTHIRD APPLICATION TEST")
    app_second(compress_DFT)


def app_fourth():
    cmap = plt.get_cmap('gray')  # To save grayscale pictures / plot grayscale images
    print("\nFOURTH APPLICATION TEST")
    print("__________________________")
    print("Description: This third application test will compare the dft and fft algorithms in the compression"
          "of the same koala image.")
    print("Expected Output: Benchmark of 3 compressions of 3 different images comparing the DFT and FFT algorithms.")
    print("Output: ")

    image_MoulinRouge = np.asarray(Image.open('MoulinRouge.jpg'))  # Import the image
    grayscale_MoulinRouge = np.dot(image_MoulinRouge[..., :3],
                                   [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    plt.imsave('MoulinRouge.jpg', grayscale_MoulinRouge, cmap=cmap)
    size_MoulinRouge = os.path.getsize('MoulinRouge.jpg') / 1000  # Get the size of the original image in kilobytes

    image_Koala = np.asarray(Image.open('Koala.jpg'))  # Import the image
    grayscale_Koala = np.dot(image_Koala[..., :3],
                             [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    plt.imsave('Koala.jpg', grayscale_Koala, cmap=cmap)
    size_Koala = os.path.getsize('Koala.jpg') / 1000  # Get the size of the original image in kilobytes

    image_Fourviere = np.asarray(Image.open('Fourviere.jpg'))  # Import the image
    grayscale_Fourviere = np.dot(image_Fourviere[..., :3],
                                 [0.2989, 0.5870, 0.1140]) / 255  # Get the Grayscale version of the image
    # Save the greyscale image to check its size and to make sure the compression is not impacted by the "imsave" method
    plt.imsave('Fourviere.jpg', grayscale_Fourviere, cmap=cmap)
    size_Fourviere = os.path.getsize('Fourviere.jpg') / 1000  # Get the size of the original image in kilobytes

    # Configure the plot by adding the input images in a first column.
    cmap = plt.get_cmap('gray')
    _, plots = plt.subplots(3, 2, figsize=(10, 8))
    plt.setp(plots, xticks=[], yticks=[])
    plots[0][0].set_title("Input 2048x2048 image, size={}KB".format(size_MoulinRouge), size=10)
    plots[0][0].imshow(grayscale_MoulinRouge, cmap, vmin=0, vmax=1)
    plots[1][0].set_title("Input 1024x1024 image, size={}KB".format(size_Koala), size=10)
    plots[1][0].imshow(grayscale_Koala, cmap, vmin=0, vmax=1)
    plots[2][0].set_title("Input 512x512 image, size={}KB".format(size_Fourviere), size=10)
    plots[2][0].imshow(grayscale_Fourviere, cmap, vmin=0, vmax=1)

    # Variables to keep track of the time taken for each compression for both DFT and FFT algorithm
    bench_FFT_result = {}
    bench_DFT_result = {}

    # 2048x2048 picture compression using FFT
    start_time = timeit.default_timer()
    compressed_MoulinRouge = compress(grayscale_MoulinRouge, 0.009)
    bench_FFT_result[2048] = timeit.default_timer() - start_time
    # Save the compressed image in the current package to get its size
    plt.imsave('compressed_MoulinRouge.jpg', compressed_MoulinRouge, cmap=cmap)
    plots[0][1].imshow(compressed_MoulinRouge, cmap, vmin=0, vmax=1)

    # 2048x2048 picture compression using DFT
    start_time = timeit.default_timer()
    compress_DFT(grayscale_MoulinRouge, 0.009)
    bench_DFT_result[2048] = timeit.default_timer() - start_time

    # 1024x1024 picture compression using FFT
    start_time = timeit.default_timer()
    compressed_Koala = compress(grayscale_Koala, 0.009)
    bench_FFT_result[1024] = timeit.default_timer() - start_time
    # Save the compressed image in the current package to get its size
    plt.imsave('compressed_Koala.jpg', compressed_Koala, cmap=cmap)
    plots[1][1].imshow(compressed_Koala, cmap, vmin=0, vmax=1)

    # 1024x1024 picture compression using DFT
    start_time = timeit.default_timer()
    compress_DFT(grayscale_Koala, 0.009)
    bench_DFT_result[1024] = timeit.default_timer() - start_time

    # 512x512 picture compression using FFT
    start_time = timeit.default_timer()
    compressed_Fourviere = compress(grayscale_Fourviere, 0.009)
    bench_FFT_result[512] = timeit.default_timer() - start_time
    # Save the compressed image in the current package to get its size
    plt.imsave('compressed_Fourviere.jpg', compressed_Fourviere, cmap='gray')
    plots[2][1].imshow(compressed_Fourviere, cmap, vmin=0, vmax=1)

    # 1024x1024 picture compression using DFT
    start_time = timeit.default_timer()
    compress_DFT(grayscale_Fourviere, 0.009)
    bench_DFT_result[512] = timeit.default_timer() - start_time
    plt.show()

    lists = sorted(bench_FFT_result.items())  # sorted by key, return a list of tuples
    N, FFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, FFT_time, label="FFT")

    lists = sorted(bench_DFT_result.items())  # sorted by key, return a list of tuples
    N, oldDFT_time = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(N, oldDFT_time, label="DFT")

    plt.title("Graph comparing the time efficiency of the DFT and FFT algorithm for the compression application.")
    plt.xlabel("Value of N (Size of the image)")
    plt.ylabel("Time taken (seconds)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # first()
    #
    # second()
    #
    # third()
    #
    # fourth()
    #
    # fifth()
    #
    # sixth()
    #
    # seventh()
    #
    # eighth()
    #
    # ninth()
    #
    # tenth()
    #
    # eleventh()
    #
    # twelfth()
    #
    # thirteenth()
    #
    # fourteenth()
    #
    # fifteenth()

    # Application
    print("The application of the FFT algorithm we chose consists of compressing a given image to reduce its size.")
    print("This is done by taking the fast fourier transform of an input image (in grayscale) and by removing "
          "(set to 0) the lowest frequencies of that fft depending on a threshold we specify. We then use the inverse "
          "fft algorithm we wrote to go generate the compressed image.")
    print("We defined a function named \"compress\" that takes as argument an image and the percentage of frequencies "
          "to keep in the fast fourier transform. For instance, compress(imageTest, 0.7) will remove 30% of all the "
          "frequencies in the fourier domain (only the lowest ones).")

    # app_first()

    # app_second()

    app_third()

    #app_fourth()

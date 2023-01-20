# Project Description
The goal of this project is to implement the Fast Fourier Transform (FFT), an alternative to the DFT algorithm that scales much more favourably for large N.
We applied the FFT algorithm to the following discrete signal processing problem: image compression. We benchmarked its perfromance and compared it to the
one of a naive DFT algorithm as a function of discrete signal size N.

# Implementation details
The Cooley-Tukey Fast Fourier Transform algorithm applies a hierarchical and recursive strategy to accelerate the computation of the DFT (and, so too, the iDFT). 
Its running time is O(𝑁log𝑁). Since the FFT operates recursively over power-of-2 reductions in the input signal, the input signal must have a power-of-2 
length. If not, in practice we can pad the input appropriately until it reaches the next power-of-2 length by using different tehcniques. For the purpose
of this project we used images with power-of-2 heights and widths.

The derivation of this FFT algorithm requires us to first notice an important property of the DFT 𝐟ˆ of a sampled input signal 𝐟 by first considering how 
the value of the complex exponential exp(−2𝜋ı𝑘𝑥/𝑁)in the DFT definition changes if we index 𝑘 past the last ((𝑁−1)𝑡ℎ index, we note that 𝑓̂[𝑎+𝑏𝑁]=𝑓̂[𝑎]. 
In other words, the sampled values 𝑓̂[⋅] of the DFT are 𝑁
N-periodic.

The FFT algorithm exploits this periodicity to accelerate computation: the DFT of a length-𝑁 sampled signal 𝑓 can be computed from the DFT of two 
length -(𝑁/2)signals 𝑓𝑒 and 𝑓𝑜 composed of the even and odd indexed values of 𝑓: 
<img width="602" alt="Screenshot 2023-01-20 at 5 36 00 PM" src="https://user-images.githubusercontent.com/72216366/213817499-44d9ff11-d7bc-438a-80d6-1ea52e81c235.png">

While applying this decomposition once would accelerate the computation of a length-𝑁 DFT by roughly a factor of two, the Cooley-Tukey algorithm 
recursively applies this decomposition until the problem is reduced to that of computing a very small DFT. In the limit, assuming that 𝑁 is a power-of-two,
the base case of the recursion simply computes a length-1 DFT2.

The results of the benchmark comparison between the FFT and DFT and the application of the algorithm to a discrete signal processing problem are 
available in the project report in the next section.

# Report 

![Project Report](https://user-images.githubusercontent.com/72216366/213817380-5ab68f63-2765-4790-a1b1-1711c8c99e7e.png)


# Description of the other data files present in the submission:

- Koala.jpg, Fourviere.jpg, MoulinRouge.jpg and VieuxLyon.jpg are the only other files present in the folder of submission. 
They consists of pictures on which we tested our fft and dft algorithms in the main.py file.

# Reference

ECSE 343 — Numerical Methods in Engineering - Winter 2022 - McGill University — Prof. Derek Nowrouzezahrai
http://www.cim.mcgill.ca/~derek/ecse343.html#collaboration&plagiarism

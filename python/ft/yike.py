import numpy as np
import matplotlib.pyplot as plt

def prepare(N, freq):
    """generate the cosine function with given number of grid and
    frequency.
    Args:
        N: int, number of points
        freq: float, frequency of the cosine function
    Returns:
        y: ndarray, the cosine function
    """
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.cos(freq*x)
    return x, y

def example2(N, freq):
    """on finite grid points, the multiplication operation should be
    carefully treated. Denote one function to be f1 and the other to
    be f2, expand them with Fourier series, we have:
    f1 = sum(a_n * exp(i*2*pi*n*x/N))
    f2 = sum(b_n * exp(i*2*pi*n*x/N))
    f1*f2 = sum(a_n * b_m * exp(i*2*pi*(n+m)*x/N))
    you can quickly realize that the frequency of the product may be
    larger than the original frequency. This is called aliasing.
    Here we illustrate this with a simple example.
    As result, you will find even if you double the grid points, the
    accuracy is still bad, this is because the functions made product
    are only sampled at the original grid points, which means, the
    information is not enough to reconstruct the product function.
    Args:
        N: int, number of points
        freq: float, frequency of the cosine function
    """
    x, y = prepare(N, freq) # this is either f1 or f2

    y2ref = y**2
    yf = np.fft.fft(y) # this is either a_n or b_n

    # dk = 1
    k = np.arange(N)


    yf2 = np.convolve(yf, yf)
    y2 = np.fft.ifft(yf2)
    print(np.linalg.norm(y2ref-y*y))
    plt.plot(x, y2ref, label='conv')
    plt.plot(x, y*y, label='direct')
    plt.show()

    return



    yfyf = np.outer(yf, yf)

    # because f1 and f2 both have components at the maximum frequency
    # denoted as qmax, then their product will have components at
    # 2*qmax. Therefore we need to double the number of points.
    yf2 = np.zeros(2*N, dtype=complex)
    for i in range(N):
        for j in range(N):
            yf2[i + j] += yfyf[i][j] / N
    # transform back to the real space
    y2 = np.fft.ifft(yf2)
    # to see what we have...
    x = np.linspace(0, 2*np.pi, 2*N, endpoint=False)
    plt.plot(x, y2ref - 1/2, label='direct')
    plt.plot(x, y2, label='FFT, doubled G-grid')
    plt.plot(x, np.fft.ifft(yf2[:N])*2, label='FFT, original G-grid')
    plt.legend()
    #plt.savefig('out.png')
    plt.show()

if __name__ == '__main__':
    example2(64, 3)


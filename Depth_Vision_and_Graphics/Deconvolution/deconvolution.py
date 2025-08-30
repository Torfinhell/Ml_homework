import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, ifftshift

def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    i, j=np.ogrid[:size, :size]
    x0, y0=(size-1)/2,(size-1)/2
    A=1/(2*np.pi*sigma**2)*np.exp(-((i-x0)**2+(j-y0)**2)/(2*sigma**2))
    return A/A.sum()


def fourier_transform(img, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    h, w = shape
    prev_h, prev_w = img.shape[:2]
    diff_h, diff_w = h - prev_h, w - prev_w

    padding = [((diff_h + 1) // 2, diff_h // 2), ((diff_w + 1) // 2, diff_w // 2)]
    new_img = np.pad(img, padding, mode="constant")
    return fft2(ifftshift(new_img))


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H[abs(H)<=threshold]=0
    H[H!=0]=1/H[H!=0]
    return H


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G=fft2(blurred_img)
    H=fourier_transform(h, blurred_img.shape)
    H_inv=inverse_kernel(H, threshold)
    return np.abs(ifft2(G*H_inv))


def wiener_filtering(blurred_img, h, K=0.00005):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G=fft2(blurred_img)
    h_fft=fourier_transform(h, blurred_img.shape)
    F=(np.conjugate(h_fft)/(np.conjugate(h_fft)*h_fft+K))*G
    return np.abs(ifft2(F))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    max_val = 255
    return 20*np.log10(max_val/np.sqrt(((img1-img2)**2).mean()))

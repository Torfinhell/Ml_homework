import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Your code here

    # Отцентруем каждую строчку матрицы
    matrix_mean = np.mean(matrix, axis=1)
    centered = matrix - matrix_mean[:, None]
    # Найдем матрицу ковариации
    cov = centered @ centered.T / matrix.shape[1]
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Посчитаем количество найденных собственных векторов
    # Сортируем собственные значения в порядке убывания
    ids = np.argsort(eigvals)[::-1]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    sorte_eigvecs = eigvecs[:, ids]
    # Оставляем только p собственных векторов
    U = sorte_eigvecs[:, :p]
    # Проекция данных на новое пространство
    proj = U.T @ centered
    return U, proj, matrix_mean


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Your code here
        U, projection, means = comp
        result_img.append(U @ projection+ means[:, None])
    return np.stack(result_img, axis=2).astype(np.int64).clip(0,255)


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = [pca_compression(img[..., j], p) for j in range(3)]
        axes[i // 3, i % 3].imshow(pca_decompression(compressed))
        axes[i // 3, i % 3].set_title(f"Кол-во компонент: {p}")

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    # Your code here
    rgb2ycbcr = np.array([
        [0.299, 0.587, 0.114], 
        [-0.1687, -0.3313, 0.5], 
        [0.5, -0.4187, -0.0813]])
    bias = np.array([0.0, 128.0, 128.0])
    ycbcr = img.reshape(-1, 3) @ rgb2ycbcr.T + bias
    return ycbcr.reshape(img.shape)


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    # Your code here
    ycbcr2rgb = np.array([
        [1.0, 0.0, 1.402], 
        [1.0, -0.34414, -0.71414], 
        [1.0, 1.77, 0.0]
        ])
    bias = np.array([0.0, 128.0, 128.0])
    rgb = np.matmul(img.reshape(-1, 3) - bias, ycbcr2rgb.T)
    return rgb.reshape(img.shape)


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    ycbcr = rgb2ycbcr(rgb_img)
    y, cb, cr = np.dsplit(ycbcr, 3)
    cb = gaussian_filter(cb.squeeze(), 10)
    cr = gaussian_filter(cr.squeeze(), 10)
    rgb = ycbcr2rgb( np.concatenate([y, cb[..., None], cr[..., None]], axis=2))
    rgb = np.clip(np.round(rgb), 0, 255).astype(np.int64)
    plt.imshow(rgb)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    ycbcr = rgb2ycbcr(rgb_img)
    y, cb, cr = np.dsplit(ycbcr, 3)
    y = gaussian_filter(y.squeeze(), 10)
    rgb = ycbcr2rgb( np.concatenate([y[..., None], cb, cr], axis=2))
    rgb = np.clip(np.round(rgb), 0, 255).astype(np.int64)
    plt.imshow(rgb)
    plt.savefig("gauss_1.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """

    # Your code here
    blurred = gaussian_filter(component, 10)
    return blurred[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    # Your code here
    N = 8
    result = np.zeros((N, N), dtype=np.float64)
    block = block.astype(np.float64)
    for u in range(N):
        for v in range(N):
            alpha_u = 1/np.sqrt(2) if u == 0 else 1.0
            alpha_v = 1/np.sqrt(2) if v == 0 else 1.0
            s = 0.0
            for x in range(N):
                for y in range(N):
                    s += block[x, y] * np.cos((2*x+1)*u*np.pi/16) * np.cos((2*y+1)*v*np.pi/16)
            result[u, v] = 0.25 * alpha_u * alpha_v * s
    return result


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    # Your code here
    return np.round(block / np.clip(quantization_matrix, 1, np.inf))


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100
    if q < 50:
        scale = 5000.0 / q
    else:
        scale = 200.0 - 2.0 * q
    
    matrix = np.floor((default_quantization_matrix * scale + 50.0) / 100.0)
    return np.clip(matrix, 0,np.inf)


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    # Your code here
    arr=[]
    i=0
    j=0
    size = block.shape[0]
    for _ in range(size*size):
        arr.append(block[i][j])
        if (i + j) % 2 == 0:
            if j < size - 1 and i > 0:
                i -= 1
                j += 1
            elif j < size - 1:
                j += 1
            else:
                i += 1
        else:
            if i < size - 1 and j > 0:
                i += 1
                j -= 1
            elif i < size - 1:
                i += 1
            else:
                j += 1
    return np.array(arr)


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    cnt_zero=0
    ans=[]
    for elem in zigzag_list:
        if(elem==0):
            cnt_zero+=1
        if(elem!=0):
            if(cnt_zero):
                ans.extend([0,cnt_zero])
                cnt_zero=0
            ans.append(elem)
    if(cnt_zero):
        ans.extend([0,cnt_zero])
    return ans
def apply_compression_to_8_by_8(image,quantizer):
    matrix_blocks=[]
    for i in range(0,image.shape[0],8):
        for j in range(0,image.shape[1],8):
            block=image[i:i+8,j:j+8]-128
            block=dct(block)
            block=quantization(block,quantizer)
            block=zigzag(block)
            block=compression(block)
            matrix_blocks.append(block)
    return matrix_blocks
def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here

    # Переходим из RGB в YCbCr
    ycbcr=rgb2ycbcr(img)
    y,cb,cr=np.dsplit(ycbcr,3)
    y=y.squeeze()
    cb=cb.squeeze()
    cr=cr.squeeze()

    # Уменьшаем цветовые компоненты
    cb=downsampling(cb)
    cr=downsampling(cr)

    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    y_blocks=apply_compression_to_8_by_8(y.squeeze(), quantization_matrixes[0])
    cb_blocks=apply_compression_to_8_by_8(cb.squeeze(), quantization_matrixes[1])
    cr_blocks=apply_compression_to_8_by_8(cr.squeeze(), quantization_matrixes[1])
    return [y_blocks,cb_blocks,cr_blocks]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    # Your code here
    ans=[]
    i=0
    while i<len(compressed_list):
        if(compressed_list[i]==0):
            ans.extend([0]*compressed_list[i+1])
            i+=2
        else:
            ans.append(compressed_list[i])
            i+=1
    return np.array(ans)


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    # Your code here
    block=np.zeros((8,8))
    i=0
    j=0
    size=8
    for elem in input:
        block[i][j]=elem
        if (i + j) % 2 == 0:
            if j < size - 1 and i > 0:
                i -= 1
                j += 1
            elif j < size - 1:
                j += 1
            else:
                i += 1
        else:
            if i < size - 1 and j > 0:
                i += 1
                j -= 1
            elif i < size - 1:
                i += 1
            else:
                j += 1
    return block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    # Your code here
    return block * np.clip(quantization_matrix,1,  np.inf)


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here
    N=8
    result=np.zeros((N,N))
    for x in range(N):
        for y in range(N):
            s=0.0
            for u in range(N):
                for v in range(N):
                    alpha_u=1/np.sqrt(2) if u==0 else 1.0
                    alpha_v=1/np.sqrt(2) if v==0 else 1.0
                    s+=alpha_u*alpha_v*block[u,v]*np.cos((2*x+1)*u*np.pi/16)*np.cos((2*y+1)*v*np.pi/16)
            result[x,y]=np.round(0.25*s)
    return result


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    # Your code here
    h,w=component.shape[:2]
    result=np.zeros((2*h,2*w,1))
    for i in range(h):
        for j in range(w):
            result[2*i:2*i+2,2*j:2*j+2]=component[i,j]
    return result

def decompress_block(block,quantization_matrix):
        decompressed=inverse_compression(block)
        block=inverse_zigzag(decompressed)
        block=inverse_quantization(block,quantization_matrix)
        block=inverse_dct(block)
        return block+128

def reconstruct_from_blocks(blocks,h,w,quantization_matrix):
    result=np.zeros((h,w))
    block_idx=0
    for i in range(0,h,8):
        for j in range(0,w,8):
            if block_idx<len(blocks):
                result[i:i+8,j:j+8]=decompress_block(blocks[block_idx],quantization_matrix)
                block_idx+=1
    return result
def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    # Your code here

    y_blocks,cb_blocks,cr_blocks=result
    h,w=result_shape[:2]
    y=reconstruct_from_blocks(y_blocks,h,w,quantization_matrixes[0])
    cb=reconstruct_from_blocks(cb_blocks,h//2,w//2,quantization_matrixes[1])
    cr=reconstruct_from_blocks(cr_blocks,h//2,w//2,quantization_matrixes[1])
    cb = upsampling(cb[...,None])
    cr = upsampling(cr[...,None])
    ycbcr = np.dstack([y, cb.squeeze(), cr.squeeze()])
    rgb=ycbcr2rgb(ycbcr)
    return np.clip(rgb,0,255).astype(np.int64)


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        y_quantization=own_quantization_matrix(y_quantization_matrix,p)
        color_quantization=own_quantization_matrix(color_quantization_matrix,p)
        matrixes=[y_quantization,color_quantization]

        compressed=jpeg_compression(img,matrixes)
        decompressed=jpeg_decompression(compressed,img.shape,matrixes)
        
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    # pca_visualize()
    # get_gauss_1()
    # get_gauss_2()
    # jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()

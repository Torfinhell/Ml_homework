import numpy as np
def get_bayer_masks(n_rows, n_cols):
    """
    :param n_rows: `int`, number of rows
    :param n_cols: `int`, number of columns

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
        containing red, green and blue Bayer masks
    """
    red_mask, green_mask, blue_mask = (np.zeros((n_rows, n_cols), dtype=bool) for _ in range(3))
    red_mask[::2,1::2]=1
    green_mask[0::2,0::2]=1
    green_mask[1::2,1::2]=1
    blue_mask[1::2,::2]=1
    return np.dstack((red_mask, green_mask, blue_mask))

def get_colored_img(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        each channel contains known color values or zeros
        depending on Bayer masks
    """
    gt_masks=get_bayer_masks(*raw_img.shape).astype(dtype="uint8")
    return raw_img[..., np.newaxis]*gt_masks


def get_raw_img(colored_img):
    """
    :param colored_img:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        colored image

    :return:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image as captured by camera
    """
    gt_masks=get_bayer_masks(colored_img.shape[0], colored_img.shape[1]).astype(dtype="uint8")
    
    return np.sum(colored_img*gt_masks, axis=-1)
    


def bilinear_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """
    h, w=raw_img.shape
    print(h, w)
    colored_img=get_colored_img(raw_img)
    sum=np.zeros((h+2, w+2, 3), dtype="uint8")
    for i in range(3):
        for j in range(3):
            sum+=np.pad(colored_img, ((i, 2-i), (j, 2-j), (0,0)), "constant")
    sum=sum[1:-1, 1:-1]
    red_mask, green_mask, blue_mask=np.dsplit(get_bayer_masks(h, w), 3)
    green_mask=green_mask.squeeze()
    sum[green_mask]//=2
    sum[~green_mask]//=4
    sum[get_bayer_masks(h, w)]=colored_img[get_bayer_masks(h, w)]
    print(sum)
    return sum
    
    

def improved_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    ...


def compute_psnr(img_pred, img_gt):
    """
    :param img_pred:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param img_gt:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric
    """
    ...


if __name__ == "__main__":
    from PIL import Image

    raw_img_path = "tests/04_unittest_bilinear_img_input/02.png"
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save("bilinear.png")

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save("improved.png")

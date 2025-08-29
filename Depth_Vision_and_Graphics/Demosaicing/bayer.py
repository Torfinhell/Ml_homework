import numpy as np
from math import log10
from numpy.lib.stride_tricks import sliding_window_view
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
    

def apply_filter(img, filter):
    """
    img and filter should have the same amount of channels
    """
    if(img.ndim==2):
        img=img[..., np.newaxis]
    if(filter.ndim==2):
        filter=filter[..., np.newaxis]
    h, w, c=img.shape
    a, b, c1 =filter.shape
    assert c1==c
    assert a%2==1 and b %2==1
    img=np.pad(img, (((a-1)//2, (a-1)//2), ((b-1)//2, (b-1)//2), (0,0)), mode="constant")
    windows = sliding_window_view(img, window_shape=(a, b), axis=(0, 1)) #( w0, h0, window1, window2, c)
    new_img=(windows*np.transpose(filter, (2, 0, 1))).sum(axis=(3,4))
    return new_img
            
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
    colored_img=get_colored_img(raw_img)
    sum=apply_filter(colored_img, np.ones((3, 3, 3)))
    red_mask, green_mask, blue_mask=np.dsplit(get_bayer_masks(h, w), 3)
    green_mask=green_mask.squeeze()
    sum[green_mask]//=2
    sum[~green_mask]//=4
    sum[get_bayer_masks(h, w)]=colored_img[get_bayer_masks(h, w)]
    return sum.astype("uint8")
    
    

def improved_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    G_at_R=np.array(#G at R locations
        [[0, 0, -1/4, 0, 0],
        [0, 0, 0, 0, 0],
        [-1/4, 0, 1, 0, -1/4],
        [0, 0, 0, 0, 0],
        [0, 0, -1/4, 0, 0]],
        dtype=np.float32
    )
    G_at_B=G_at_R#G at B locations
    R_at_G_in_R_B=np.array(#R at green in R row, B column
        [[0, 0, 1/10, 0, 0],
        [0, -1/5, 0, -1/5, 0],
        [-1/5, 0, 1, 0, -1/5],
        [0, -1/5, 0, -1/5, 0],
        [0, 0, 1/10, 0, 0]],
        dtype=np.float32
    )
    R_at_G_in_B_R=R_at_G_in_R_B.T#R at green in B row, R column
    R_at_B=np.array( #R at blue in B row, B column
        [[0,0,-3/12, 0,0],
        [0, 0, 0, 0,0],
        [-3/12, 0, 1, 0, -3/12],
        [0, 0, 0, 0, 0],
        [0, 0, -3/12, 0, 0]],
        dtype=np.float32
    )
    B_at_G_in_B_R=G_at_R#B at green in B row, R column
    B_at_G_in_R_B=G_at_R.T#B at green in R row, B column
    B_at_R=R_at_B#B at red in R row, R column
    h, w=raw_img.shape
    colored_img=get_colored_img(raw_img).astype(np.float32)
    red_mask, green_mask, blue_mask=np.dsplit(get_bayer_masks(h, w), 3)
    green_mask=green_mask.squeeze()
    red_mask=red_mask.squeeze()
    green_mask=green_mask.squeeze()
    alpha=1/2
    beta=5/8
    gamma=3/4
    i, j=np.ogrid[:h, :w]
    mask_at_R_B=(i%2+2*(j%2))==2
    mask_at_B_R=(i%2+2*(j%2))==1
    colored_img[1]+=alpha* apply_filter((colored_img*red_mask[..., np.newaxis])[...,0], G_at_R)*red_mask
    colored_img[1]+=alpha* apply_filter((colored_img*blue_mask[..., np.newaxis])[...,2],G_at_B)*blue_mask
    colored_img[0]+=beta*apply_filter((colored_img*mask_at_R_B[..., np.newaxis])[...,1], R_at_G_in_R_B)*mask_at_R_B
    colored_img[0]+=beta* apply_filter((colored_img*mask_at_B_R[..., np.newaxis])[...,1], R_at_G_in_B_R)*mask_at_B_R
    colored_img[0]+=beta* apply_filter((colored_img*blue_mask[..., np.newaxis])[...,2], R_at_B)*blue_mask
    colored_img[2]+=gamma* apply_filter((colored_img*mask_at_B_R[..., np.newaxis])[...,1], B_at_G_in_B_R)*mask_at_B_R
    colored_img[2]+=gamma* apply_filter((colored_img*mask_at_R_B[..., np.newaxis])[...,1], B_at_G_in_R_B)*mask_at_R_B
    colored_img[2]+=beta* apply_filter((colored_img*red_mask[..., np.newaxis])[...,0], B_at_R)*red_mask
    return colored_img

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
    img_pred=img_pred.astype("int32")
    img_gt=img_gt.astype("int32")
    h, w, c=img_pred.shape
    mse=(1/(h*w*c)) * ((img_pred-img_gt)**2).sum()
    if(mse==0):
        raise ValueError("The images are identical. PSNR cannot be computed")
    return 10 * log10((img_gt.max()**2)/mse)

if __name__ == "__main__":
    from PIL import Image

    raw_img_path = "tests/04_unittest_bilinear_img_input/02.png"
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save("bilinear.png")

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save("improved.png")

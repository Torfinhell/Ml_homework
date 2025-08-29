import numpy as np
from math import sqrt

# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.
def extract_channel_plates(raw_img, crop):
    h, w=raw_img.shape
    small_h=h//3
    up_coords = (np.array([0, 0]), np.array([small_h, 0]), np.array([2*small_h, 0]))
    down_coords = (np.array([small_h, w]), np.array([2*small_h, w]), np.array([3*small_h, w]))
    if crop:
        up_coords=tuple(np.array([coord[0]+(small_h//10) , coord[1]+w//10]) for coord in up_coords)
        down_coords=tuple(np.array([coord[0]-(small_h//10), coord[1]-w//10]) for coord in down_coords)
    unaligned_rgb = (raw_img[up_coords[0][0]:down_coords[0][0], up_coords[0][1]:down_coords[0][1]], raw_img[up_coords[1][0]:down_coords[1][0], up_coords[1][1]:down_coords[1][1]], 
    raw_img[up_coords[2][0]:down_coords[2][0], up_coords[2][1]:down_coords[2][1]])
    ans_unaligned_rgb=(unaligned_rgb[2], unaligned_rgb[1], unaligned_rgb[0])#because in bgr order
    ans_up_coords=(up_coords[2], up_coords[1], up_coords[0])
    return ans_unaligned_rgb, ans_up_coords

def Mse(img1,img2):
    assert img1.shape==img2.shape
    h, w=img1.shape
    return  ((img1-img2)**2).sum()/(h*w)
def Norm_cross_cor(img1, img2):
    assert img1.shape==img2.shape
    return (img1*img2).sum()/sqrt((img1**2).sum()*(img2**2).sum())
def resize_by_2(img):
    h, w=img.shape
    h-=h%2
    w-=w%2
    return (img[:h:2, :w:2]+img[:h:2, 1:w:2]+img[1:h:2, :w:2]+img[1:h:2, 1:w:2])/4
def calculate_metric_shift(img1, img2, shift, chosen_metric=Mse): #calculates metric between  img1 and shifted img2
    assert img1.shape==img2.shape
    x0, x1=max(0, shift[0]), min(img1.shape[0], img1.shape[0]+shift[0])
    y0, y1=max(0, shift[1]), min(img1.shape[1], img1.shape[1]+shift[1])
    shift=(-shift[0], -shift[1])
    x2, x3=max(0, shift[0]), min(img1.shape[0], img1.shape[0]+shift[0])
    y2, y3=max(0, shift[1]), min(img1.shape[1], img1.shape[1]+shift[1])
    return chosen_metric(img1[x2:x3, y2:y3], img2[x0:x1, y0:y1])
def find_relative_shift_pyramid(img_a, img_b, window_size=15,chosen_metric=Norm_cross_cor):
    assert img_a.shape==img_b.shape
    resized_images_a=[img_a]
    resized_images_b=[img_b]
    iterations=0
    while(resized_images_a[0].shape[0]>500 or resized_images_a[0].shape[1]>500 or iterations==0):
        resized_images_a.insert(0, resize_by_2(resized_images_a[0]))
        resized_images_b.insert(0, resize_by_2(resized_images_b[0]))
        iterations+=1

    shift=(0,0)
    optimal_shift=shift
    window_size=window_size//2+2
    for _ in range(iterations+1):
        img_a=resized_images_a.pop(0)
        img_b=resized_images_b.pop(0)
        optimal_shift=shift
        optimal_metric=calculate_metric_shift(img_a, img_b, optimal_shift, chosen_metric)
        for i in range(-window_size, window_size+1):
            for j in range(-window_size, window_size+1):
                check_shift=(shift[0]+i, shift[1]+j)
                metric=calculate_metric_shift(img_a, img_b, check_shift, chosen_metric)
                if((metric<optimal_metric and chosen_metric==Mse) or (metric>optimal_metric and chosen_metric==Norm_cross_cor)):
                    optimal_metric=metric
                    optimal_shift=check_shift
        shift=(optimal_shift[0]*2, optimal_shift[1]*2)
        window_size=1
    return np.array(list(optimal_shift))    

def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    r_to_g = find_relative_shift_fn(crops[0], crops[1])+crop_coords[1]-crop_coords[0]
    b_to_g = find_relative_shift_fn(crops[2], crops[1])+crop_coords[1]-crop_coords[2]
    return r_to_g, b_to_g


def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    red_coords, green_coords, blue_coords=channel_coords
    red_coords+=r_to_g
    blue_coords+=b_to_g
    red_coords-=green_coords
    blue_coords-=green_coords
    green_coords-=green_coords
    img_shape=channels[0].shape
    red_channel, green_channel, blue_channel=channels
    x_green_0, y_green_0=np.dstack((red_coords, green_coords, blue_coords)).max(axis=-1).squeeze()
    x_green_1, y_green_1=np.dstack((red_coords+img_shape, green_coords+img_shape, blue_coords+img_shape)).min(axis=-1).squeeze()
    green_channel=green_channel[x_green_0:x_green_1, y_green_0:y_green_1]
    green_coords-=red_coords
    blue_coords-=red_coords
    red_coords-=red_coords
    x_red_0, y_red_0=np.dstack((red_coords, green_coords, blue_coords)).max(axis=-1).squeeze()
    x_red_1, y_red_1=np.dstack((red_coords+img_shape, green_coords+img_shape, blue_coords+img_shape)).min(axis=-1).squeeze()
    red_channel=red_channel[x_red_0:x_red_1, y_red_0:y_red_1]
    green_coords-=blue_coords
    red_coords-=blue_coords
    blue_coords-=blue_coords
    x_blue_0, y_blue_0=np.dstack((red_coords, green_coords, blue_coords)).max(axis=-1).squeeze()
    x_blue_1, y_blue_1=np.dstack((red_coords+img_shape, green_coords+img_shape, blue_coords+img_shape)).min(axis=-1).squeeze()
    blue_channel=blue_channel[x_blue_0:x_blue_1, y_blue_0:y_blue_1]
    aligned_img = np.dstack((red_channel, green_channel, blue_channel))
    return aligned_img


def find_relative_shift_fourier(img_a, img_b):
    C=np.fft.ifft2(np.fft.fft2(img_a)*np.conj(np.fft.fft2(img_b)))
    flat_idx=np.argmax(np.real(C))
    u, v=np.unravel_index(flat_idx, C.shape)
    h, w=C.shape
    u=(u+h//2)%h-h//2
    v=(v+w//2)%w-w//2
    return -np.array([u, v])


if __name__ == "__main__":
    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)

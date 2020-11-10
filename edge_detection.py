#!/usr/local/bin/python3.8

from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import sys

def image_to_numbers(img_file):
    img = Image.open(img_file)
    img.load()
    data = np.asarray(img, dtype="int16")
    return data

def show_image(data):
    array_dimension = len(np.shape(data))
    if array_dimension == 2:
        img = Image.fromarray(data.astype("uint8"), "L")
    else:
        img = Image.fromarray(data.astype("uint8"), 'RGB')

    img.show()

def adaptive_median_filter(x):
    # Image height and width
    x_height, x_width, _ = np.shape(x)

    # Window height and width, which will be slided over the image.
    # max_size is the maximum window size during adaptive median filtering.
    w_height = 3
    w_width = 3
    max_size = 9
    filtered_img = np.zeros((x_height,x_width,3))

    # Slide the window over the image, and apply adaptive median filter
    # as described in the paper. I changed variable names to something more
    # descriptive but original ones are in comments above.
    for row in range(x_height - w_height):
        for column in range(x_width - w_width):
            for color in range(3):
                temp_height = w_height
                temp_width = w_width
                while True:
                    # Zmin, Zmax and Zmed
                    w_min = x[row:row+temp_height, column:column+temp_width, color].min()
                    w_max = x[row:row+temp_height, column:column+temp_width, color].max()
                    w_med = np.median(x[row:row+temp_height, column:column+temp_width, color])

                    med_min_diff = w_med - w_min
                    med_max_diff = w_med - w_max

                    # A1 > 0 AND A2 < 0
                    if med_min_diff > 0 and med_max_diff < 0:
                        # B1 and B2
                        org_min_diff = x[row, column, color] - w_min
                        org_max_diff = x[row, column, color] - w_max

                        # B1 > 0 and B2 < 0
                        if org_min_diff > 0 and org_max_diff < 0:
                            filtered_img[row, column, color] = x[row, column, color]
                            break
                        else:
                            filtered_img[row, column, color] = w_med
                            break
                    else:
                        # Increase the window size
                        temp_height += 2
                        temp_width += 2

                    # Sxy < Smax, I have it reverse here due to structure of this function,
                    # but functionally the same
                    if temp_height >= max_size:
                        filtered_img[row, column, color] = x[row, column, color]
                        break

    return filtered_img

def transform_pixels(x):
    # To reduce computational overhead, transform RGB values to
    # single valued attributes. Pixel(i,j) = 2*red(i,j)+3*green(i,j)+4*blue(i,j).
    conv3to1 = 2*x[:,:,0] + 3*x[:,:,1] + 4*x[:,:,2]

    return conv3to1

def directional_color_difference(x):
    x_height, x_width = np.shape(x)
    filtered_img = np.zeros((x_height, x_width))

    # Define masks which are slided over the image
    # H_mask = Horizontal, V_mask = Vertical
    # DD_mask = Descending diagonal, AD_mask = Ascending diagonal
    # Masks are already flipped horizontally and vertically from
    # what they appeared in the paper, so mask convoluted with a window
    # is just sum of the elementwise matrix product
    H_mask = np.array([[0,0,0],[-4,0,4],[0,0,0]])
    V_mask = np.array([[0,-4,0],[0,0,0],[0,4,0]])
    DD_mask = np.array([[0,0,-4],[0,0,0],[4,0,0]])
    AD_mask = np.array([[4,0,0],[0,0,0],[0,0,-4]])

    w_height = 3 # window height
    w_width = 3 # window width

    for row in range(x_height - w_height):
        for column in range(x_width - w_width):
            window = x[row:row+w_height, column:column+w_width]

            # Notice all windows and masks are 3x3, as defined above, and
            # window * mask is elementwise product, not traditional matrix
            # multiplication. As explained above, summing the values of the elementwise products
            # give us convolution
            H_diff = np.sum(window * H_mask)
            V_diff = np.sum(window * V_mask)
            DD_diff = np.sum(window * DD_mask)
            AD_diff = np.sum(window * AD_mask)

            max_diff = max([H_diff, V_diff, DD_diff, AD_diff])
            filtered_img[row+1,column+1] = max_diff

    return filtered_img

def thresholding(x):
    # According to the paper 1.2 * avg directional difference
    # is a good threshold, however trying out different values for
    # the first term might help in some cases
    T = 1.2 * x.mean()

    return (x < T) * 255

def thinning(x):
    x_height, x_width = np.shape(x)
    filtered_img = np.zeros((x_height, x_width))

    # Define masks which are slided over the image
    # H_mask = Horizontal, V_mask = Vertical
    # DD_mask = Descending diagonal, AD_mask = Ascending diagonal
    H_mask = np.array([[0,0,0],[-1,2,-1],[0,0,0]])
    V_mask = np.array([[0,-1,0],[0,2,0],[0,-1,0]])

    w_height = 3 # window height
    w_width = 3 # window width

    # Slide two masks over the image and perform convolution operations to
    # get thinner edges
    for row in range(x_height - w_height):
        for column in range(x_width - w_width):
            window = x[row:row+w_height, column:column+w_width]
            convolution_H = convolve2d(window, H_mask, mode="same")
            convolution_V = np.sum(convolve2d(convolution_H, V_mask, mode="same"))

            filtered_img[row, column] = convolution_V

    return filtered_img

if __name__ == "__main__":
    # Image we are filtering
    try:
        img_file = sys.argv[1]
        img_data = image_to_numbers(img_file)
    except IndexError:
        print("Usage: ./edge_detection.py <path_to_image>")
        sys.exit(1)
    except FileNotFoundError:
        print("Check file path and try again")
        sys.exit(1)

    # Apply adaptive median filter
    print("Applying adaptive median filter...")
    img_data = adaptive_median_filter(img_data)
    show_image(img_data)

    # Transformed pixel value calculation
    print("Transforming pixel values...")
    img_data = transform_pixels(img_data)

    # Calculate maximum directional color difference
    print("Applying directional masks...")
    img_data = directional_color_difference(img_data)
    show_image(img_data)

    # Threshold image
    print("Thresholding...")
    img_data = thresholding(img_data)
    show_image(img_data)

    # Thin image
    print("Thinning the edges...")
    img_data = thinning(img_data)
    show_image(img_data)
    print("Filtering done!")

from collections import deque

import matplotlib.pyplot as plt
import cv2
import numpy as np

IMAGE_OUTPUT_DIR = 'output_images/'
# BIRD_VIEW_MATRIX = None
# BIRD_INVERSE_VIEW_MATRIX = None


def show_pair_image(left_image, left_caption='', right_image=None, right_caption='', save_file_name=None, save=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
    f.tight_layout()
    ax1.imshow(left_image)
    ax1.set_title(left_caption, fontsize=30)
    ax2.imshow(right_image)
    ax2.set_title(right_caption, fontsize=30)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if save:
        plt.savefig(IMAGE_OUTPUT_DIR + save_file_name)
    else:
        plt.show()


def show_image(img, title=None, cmap=None, save_file_name=None, save=False):
    if title is not None:
        plt.title(title)
    plt.imshow(img, cmap)
    if save:
        plt.savefig(IMAGE_OUTPUT_DIR + save_file_name)
    else:
        plt.show()


def show_image_diff(img1, img2):
    changes = img1 - img2
    plt.imshow(changes)
    plt.show()


def get_birdview(image):
    image_size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, get_birdview_matrix()[0], image_size, flags=cv2.INTER_LINEAR)


def get_birdview_revserse(image):
    image_size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, get_birdview_matrix()[1], image_size, flags=cv2.INTER_LINEAR)


def get_birdview_matrix():
    if get_birdview_matrix.BIRD_VIEW_MATRIX is not None:
        return get_birdview_matrix.BIRD_VIEW_MATRIX, get_birdview_matrix.BIRD_INVERSE_VIEW_MATRIX

    # this perspective transform is fixed for 1280x720 center view camera.
    # Finally, I choose center +- point according to curvature values by offset range
    offset = 74
    src = np.float32([[640-offset, 470], [640+offset, 470],
                      [1080, 720], [200, 720]])

    offset_bird = 320
    dst = np.float32([[640-offset_bird, 0], [640+offset_bird, 0],
                      [640+offset_bird, 720], [640-offset_bird, 720]])

    get_birdview_matrix.BIRD_VIEW_MATRIX = cv2.getPerspectiveTransform(src, dst)
    get_birdview_matrix.BIRD_INVERSE_VIEW_MATRIX = cv2.getPerspectiveTransform(dst, src)
    return get_birdview_matrix.BIRD_VIEW_MATRIX, get_birdview_matrix.BIRD_INVERSE_VIEW_MATRIX

get_birdview_matrix.BIRD_VIEW_MATRIX = None
get_birdview_matrix.BIRD_INVERSE_VIEW_MATRIX = None


def sobel_binary(undist, threshold=(20, 100)):
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    sxbinary = threshold_filter(scaled_sobel, threshold)
    return sxbinary

def sobel_mono_binary(mono_image, threshold=(20, 100)):
    # Sobel x
    sobelx = cv2.Sobel(mono_image, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = threshold_filter(scaled_sobel, threshold)
    return sxbinary


def threshold_filter(mono_image, threshold=(0, 255)):
    binary = np.zeros_like(mono_image)
    binary[(mono_image >= threshold[0]) & (mono_image <= threshold[1])] = 1
    return binary


def hls_binary(undist, threshold=(170, 255)):
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = threshold_filter(s_channel, threshold)
    return s_binary


def get_binary_lane(undist):
    sxbinary = sobel_binary(undist)
    s_binary = hls_binary(undist)

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def get_binary_lane2(undist):
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)

    sxbinary = sobel_binary(undist)
    r_binary = sobel_mono_binary(undist[:, :, 0])
    g_binary = sobel_mono_binary(undist[:, :, 1])
    b_binary = sobel_mono_binary(undist[:, :, 2])

    l_binary = sobel_mono_binary(hls[:, :, 1])
    s_binary = sobel_mono_binary(hls[:, :, 2])

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1) | (l_binary == 1) |\
                    (r_binary == 1)| (g_binary == 1)| (b_binary == 1) ] = 1
    return combined_binary


def detect_lane_line(binary_warped):
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 3:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[200:midpoint]) + 200
    rightx_base = np.argmax(histogram[midpoint:-200]) + midpoint

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 100
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, out_img


MAX_QUEUE_BUFFER = 6


def find_lanefit(binary_warped):
    if len(find_lanefit.left_fit) < 1:
        left_fit, right_fit, _ = detect_lane_line(binary_warped)
        find_lanefit.left_fit.append(left_fit)
        find_lanefit.right_fit.append(right_fit)
    else:
        mean_left_fit = np.mean(find_lanefit.left_fit, axis=0)
        mean_right_fit = np.mean(find_lanefit.right_fit, axis=0)
        left_fit, right_fit = \
            get_related_lane( \
                binary_warped, mean_left_fit, mean_right_fit)

        diff_left = np.absolute(mean_left_fit - left_fit)
        diff_right = np.absolute(mean_right_fit - right_fit)

        thres = np.array([1e-4, 1e-1, 30], dtype=np.float32)

        find_lanefit.diffs_left = diff_left
        find_lanefit.diffs_right = diff_right

        if np.all(thres > diff_left) and np.all(thres > diff_right):
            find_lanefit.left_fit.append(left_fit)
            find_lanefit.right_fit.append(right_fit)
        else:
            left_fit, right_fit, _ = detect_lane_line(binary_warped)
            find_lanefit.left_fit.append(left_fit)
            find_lanefit.right_fit.append(right_fit)

    mean_left_fit = np.mean(find_lanefit.left_fit, axis=0)
    mean_right_fit = np.mean(find_lanefit.right_fit, axis=0)
    return mean_left_fit, mean_right_fit


find_lanefit.left_fit = deque(maxlen=MAX_QUEUE_BUFFER)
find_lanefit.right_fit = deque(maxlen=MAX_QUEUE_BUFFER)

find_lanefit.diffs_left = None
find_lanefit.diffs_right = None



def get_lane_curvature(shape, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, shape[0] - 1, shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 15 / 450  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 650  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_fitx, right_fitx, left_curverad, right_curverad


def get_lane_shape(binary_warped, left_fitx, right_fitx):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    right_line_pts = np.hstack((left_line_window, right_line_window))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    return window_img

def get_related_lane(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 50

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def plot_images(imgs, titles, cmap='gray', figsize=(24, 9)):
    nimgs = len(imgs)
    f, axes = plt.subplots(1, nimgs, figsize=figsize)
    f.tight_layout()
    for i in range(nimgs):
        axes[i].imshow(imgs[i], cmap=cmap)
        axes[i].set_title(titles[i], fontsize=25)


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def compose_image(main_img, sub_images=None):
    cols = 3
    rows = 3
    width = 1260 // cols
    height = 960 // rows
    compose = np.zeros((height*rows, width*cols, 3), dtype=np.uint8)
    compose[0:height*2, 0:width*2] = cv2.resize(main_img, (width*2, height*2))

    for i, img in enumerate(sub_images):
        if i < 2:
            compose[height * i:height * (i + 1), width * 2: width * 3] = cv2.resize(img, (width, height))
        elif i < 5:
            compose[height * 2:height * 3, width * (i - 2):width * (i - 1)] = cv2.resize(img, (width, height))
        else:
            compose[height * 3:height * 4, width * (i - 5):width * (i - 4)] = cv2.resize(img, (width, height))

    return compose

def extend_channel(single_channel, isBinary=True):
    result = np.dstack((single_channel, single_channel, single_channel))
    if isBinary:
        result *= 255
    return result


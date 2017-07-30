from cal_camera import get_camera_cal, undistort
from utils import *
import glob
import cv2
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np


TEST_IMAGE_DIR = './test_images'

if __name__ == '__main__':
    images = glob.glob(TEST_IMAGE_DIR + '/*.jpg')
    print(images)
    mtx, dist = get_camera_cal()

    for filename in images:
        print(filename)
        image = misc.imread(filename)
        undist = undistort(image, mtx, dist)

        # 여기부터 라인 엣지 따기
        bird_view = get_birdview(undist)
        hls = cv2.cvtColor(bird_view, cv2.COLOR_RGB2HLS)

        binary_warped = get_binary_lane2(bird_view)

        test = sobel_binary(undist)
        # test2 = hls_binary(undist)
        #

        #
        # plot_images((hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]), ('H', 'L', 'S'))
        # plot_images((test, test2, hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]), (('sobel', 'hls-s', 'H', 'L', 'S')))
        # plt.show()

        # 오케이.. 일단 컬러로도 표현해보았고.. 믹스된 채널도 출력됐다.
        # get birdview from combined_binary
        # binary_warped = get_birdview(lane_binary)


        # 자 바이너리 warp된 이미지를 가지고 차선을 찾아가자.

        left_fit, right_fit, out_img = detect_lane_line(binary_warped)

        left_fitx, right_fitx, left_curverad, right_curverad = get_lane_curvature(binary_warped.shape, left_fit, right_fit)

        # print(left_curverad, '\t', right_curverad)


        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()
        # 잠시 꺼두고


        margin = 100

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        right_line_pts = np.hstack((left_line_window, right_line_window))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        # result = cv2.addWeighted(window_img, 1, window_img, 0.3, 0)

        left_lines = np.array(np.dstack((left_fitx, ploty)).reshape((-1, 2)), np.int32)
        window_img = cv2.polylines(window_img, [left_lines], True, (255, 0, 0))
        right_lines = np.array(np.dstack((right_fitx, ploty)).reshape((-1, 2)), np.int32)
        window_img = cv2.polylines(window_img, [right_lines], True, (255, 0, 0))

        result = get_birdview_revserse(window_img)

        result = cv2.addWeighted(undist, 1, result, 0.3, 0)

        sxbinary = sobel_binary(undist)
        r_binary = sobel_mono_binary(undist[:, :, 0])
        g_binary = sobel_mono_binary(undist[:, :, 1])
        b_binary = sobel_mono_binary(undist[:, :, 2])
        l_binary = sobel_mono_binary(hls[:, :, 1])
        s_binary = sobel_mono_binary(hls[:, :, 2])

        plot_images((sxbinary, r_binary, g_binary, b_binary, l_binary, s_binary),
                    ('Sobel X', 'R Sobel', 'G Sobel', 'B Sobel', 'L Sobel', 'S Sobel'))
        plt.show()


        lane_shape = weighted_img(bird_view, window_img, 0.4)


        plot_images((binary_warped, out_img, lane_shape, result ), ('Binary Filter', 'Search Window', 'Lane Shape', 'Final'))
        plt.show()


    print('hold break')



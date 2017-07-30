from cal_camera import get_camera_cal, undistort
from utils import *
import cv2
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from moviepy.editor import VideoFileClip


def proc_pipe(image):

    # get camera calibration matrix
    mtx, dist = get_camera_cal()

    # get calibrated image
    undist = undistort(image, mtx, dist)

    # get binary lane by filter combination
    lane_binary = get_binary_lane2(undist)

    # get warped binary
    binary_warped = get_birdview(lane_binary)

    # find lane fit
    mean_left_fit, mean_right_fit = find_lanefit(binary_warped)

    # calculate fitx and curverad from the lane fit
    left_fitx, right_fitx, left_curverad, right_curverad = \
        get_lane_curvature(binary_warped.shape, mean_left_fit, mean_right_fit)

    # draw lane shape on the warped view
    lane_shape = get_lane_shape(binary_warped, left_fitx, right_fitx)

    # overlay lane shape on the main view
    main_view = weighted_img(undist, get_birdview_revserse(lane_shape), α=0.4)

    # get birdview for presentation
    warped = get_birdview(undist)

    # prepare more debug image
    binary_mixed = weighted_img(warped, extend_channel(binary_warped), α=0.4)
    lane_shape_mixed = weighted_img(warped, lane_shape, α=0.4)

    # Calculate Car Position from Center
    position = (left_fitx[-1] + right_fitx[-1])//2
    to_the_left_from_center = (640 - position) * 3.7 / 650

    # Calculate curverad
    curverad = int((left_curverad + right_curverad) / 2)

    cv2.putText(main_view, 'Radius of Curvature %5d m' % curverad, (50, 50),
                fontFace=2, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.putText(main_view, 'To the Left from Center %2.3f m' % to_the_left_from_center, (50, 90),
                fontFace=2, fontScale=1, color=(255, 255, 255), thickness=2)

    final = compose_image(main_view,
                          (lane_shape_mixed, binary_mixed,
                           undist,
                           extend_channel(lane_binary),
                           extend_channel(get_birdview(lane_binary))))

    return final

if __name__ == '__main__':
    video_file = 'project_video.mp4'
    # video_file = 'challenge_video.mp4'
    video_output = 'output_images/' + video_file
    read_clip = VideoFileClip(video_file, audio=False)
    white_clip = read_clip.fl_image(proc_pipe)
    white_clip.write_videofile(video_output, audio=False)





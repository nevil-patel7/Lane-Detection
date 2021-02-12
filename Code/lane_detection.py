from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from functions import *
# global axs, fig,out, out_plt
initial_frame_num = 0
cur_frame = initial_frame_num
frame, video = ReadFrame(initial_frame_num)
if make_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    frame_size = (1392, 512)
    file = 'OUTPUT_Problem2_Data' + str(data_set) + '.mp4'
    fps_out = 15
    out = cv2.VideoWriter(file, fourcc, fps_out, frame_size)
    if data_set == 2:
        file = 'OUTPUT_Problem2_Data' + str(data_set) + '.mp4'
        fps_out = 30
        out = cv2.VideoWriter(file, fourcc, fps_out, (1280,720))

    print('Processing and creating video..')

while video:

    # Warp the image
    top_down_image = Warp(H, frame, crop_height, crop_width)

    # Prepare the Image, Edge detection, create new image to fill with contours
    hsv_binary_image, hsv_threshs = ConvertToBinary(top_down_image)

    # Find peaks and detect Lanes
    right_peak, left_peak, found_left_lane, found_right_lane, inds, count = DetectPeaks(hsv_binary_image)

    # Fit Lines using line fitting based on DetectPeaks()
    left_lane_coeffs, right_lane_coeffs = FindLanelines(left_peak, right_peak, found_left_lane, found_right_lane, inds)

    # Overlay the Lane and Lane Lines On Original Frame
    lane_image, overlay = CreateOverlay(left_lane_coeffs, right_lane_coeffs, frame)

    print("Processing Frame: " + str(cur_frame))
    if make_video:
        out.write(overlay)
    else:
        cv2.imshow("Lane Overlay", overlay)
        cv2.waitKey(1)

    if cv2.waitKey(1) == ord('q'):
        break

    count += 1
    cur_frame += 1
    frame, video = ReadFrame(cur_frame)

if make_video:
    out.release()

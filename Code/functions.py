import numpy as np
import cv2
from scipy import signal
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


make_video =True
data_set = 1
#Get USer input for dataset
def GetUserInput():
    global make_video
    global data_set
    while(1):
        data_set = int(input("Choose dataset ( 1 or 2 )"))
        make_video = int(input("Make video output file ? (1:Yes     0:No"))
        if data_set > 0 and data_set <= 2:
            break
GetUserInput()
dst_size = (500, 500)
initial_frame_num = 0
count = 0
crop_width = dst_size[0]
crop_height = dst_size[1]


if data_set == 1:
    threshold = [[(0, 0, 220), (255, 49, 255)]]
    # region of road - determined experimentally
    corners_source = [(585, 275), (715, 275), (950, 512), (140, 512)]
    K = np.array([[ 9.037596e+02, 0.000000e+00, 6.957519e+02], [0.000000e+00, 9.019653e+02, 2.242509e+02],[ 0.000000e+00, 0.000000e+00, 1.000000e+00]])
    dist = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])
    errorbound = 60

elif data_set == 2:
    K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                  [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02])
    threshold = [[(0, 56, 100), (255, 255, 255)], [(0, 0, 190), (255, 255, 255)]]
    # region of road - determined experimentally
    corners_source = [(610, 480), (720, 480), (960, 680), (300, 680)]
    errorbound = 30
else:
    print("Invalid input. Exiting...")
    exit()

points_source = np.float32(corners_source).reshape(-1, 1, 2)  # change form for homography computation
corners_dest = [(0.2 * crop_width, 0), (0.8 * crop_width, 0), (0.8 * crop_width, crop_height),
               (0.2 * crop_width, crop_height)]
points_dest = np.float32(corners_dest).reshape(-1, 1, 2)  # change form for homography computation
H = cv2.findHomography(points_dest, points_source)[0]



def ReadFrame(frame_num):
    video=True
    if data_set == 1:
        filepath = r"data_1/data/"
        imagepath = filepath + ('0000000000' + str(frame_num))[-10:] + '.png'
        frame = cv2.imread(imagepath)
        if frame is None:
            video = False
        frame = frame

    elif data_set == 2:
        videopath = r"data_2/challenge_video.mp4"
        video = cv2.VideoCapture(videopath)
        # move the video to the start frame and adjust the counter
        video.set(1, frame_num)
        istherevideo, frame = video.read()
        # istherevideo returns false if video not found, if found 'frame' reads the video frame
        if istherevideo:
            frame = frame
        else:
            video = False
    return frame, video


def preprocess(frame):
    undistorted_img = cv2.undistort(frame, K, dist)
    blurr = cv2.GaussianBlur(undistorted_img, (3, 3), 1)
    # frame = cv2.medianBlur(frame,7)
    return blurr

def Warp(H, src, h, w):
    # create indices of the destination image and linearize them

    indy, indx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    # warp the coordinates of src to those of true_dst
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1] / map_ind[-1]
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    warped_image = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)
    return warped_image


def ConvertToBinary(top_down_image):
    # Convert the image to HSV space
    hsv_image = cv2.cvtColor(top_down_image, cv2.COLOR_BGR2HSV)

    # Thresh the image based on the HSV max/min values
    thresholded = []
    for thresh in threshold:
        thresholded.append(cv2.inRange(hsv_image, thresh[0], thresh[1]))

    # print(thresholded)
    if len(thresholded) == 1:
        hsv_binary_image = thresholded[0]

    else:
        hsv_binary_image = thresholded[0]

        for i in range(len(thresholded) - 1):
            hsv_binary_image = cv2.bitwise_or(hsv_binary_image, thresholded[i+1])
    if(data_set==2):
        hsv_binary_image=cv2.GaussianBlur(hsv_binary_image,(15,15),0)
    return hsv_binary_image, thresholded

hist_peak_r = 0
hist_peak_l = 0
def DetectPeaks(hsv_binary_image):
    global count
    global hist_peak_r,hist_peak_l
    right_lane = False
    left_lane = False
    inds=np.nonzero(hsv_binary_image)

    # Create a histogram
    num_pixels, bins = np.histogram(inds[1], bins=crop_width, range=(0, crop_width))

    # Find hist_peak_all in histogram
    hist_peak_all = signal.find_peaks_cwt(num_pixels, np.arange(1, 50))

    if len(hist_peak_all) == 0:  # No hist_peak_all detected

        hist_peak_r = hist_peak_r
        hist_peak_l = hist_peak_l
        right_lane = False
        left_lane = False

    elif len(hist_peak_all) == 1:  # only one peak detected
        if hist_peak_all[0] >= crop_width / 2 and abs(hist_peak_all[0] - hist_peak_r) < errorbound:
            hist_peak_r = hist_peak_all[0]
            hist_peak_l = hist_peak_l
            right_lane = True
            left_lane = False
        elif hist_peak_all[0] <= crop_width / 2 and abs(hist_peak_all[0] - hist_peak_l) < errorbound:
            hist_peak_l = hist_peak_all[0]
            hist_peak_r = hist_peak_r
            right_lane = False
            left_lane = True
        else:
            hist_peak_r = hist_peak_r
            hist_peak_l = hist_peak_l
            right_lane = False
            left_lane = False
    # Multiple hist_peak_all Detected
    else:
        # Find the value associated with the peak
        peak_vals = []
        for peak in hist_peak_all:
            peak_vals.append(num_pixels[peak])

        # Find the two highest hist_peak_all
        max1_ind = peak_vals.index(max(peak_vals))
        temp = peak_vals.copy()
        temp[max1_ind] = 0
        max2_ind = peak_vals.index(max(temp))
        big_hist_peak_all = [hist_peak_all[max1_ind], hist_peak_all[max2_ind]]
        big_hist_peak_all.sort()

        if count == 0:
            hist_peak_l = big_hist_peak_all[0]
            hist_peak_r = big_hist_peak_all[1]
            left_lane = True
            right_lane = True
        else:
            found_hist_peak_l = False
            found_hist_peak_r = False
            for peak in hist_peak_all:
                if abs(peak - hist_peak_l) <= errorbound:
                    found_hist_peak_l = True
                    hist_peak_l = peak
                if abs(peak - hist_peak_r) <= errorbound:
                    found_hist_peak_r = True
                    hist_peak_r = peak
            # print("found left peak",found_hist_peak_l, "Found right PEak", found_hist_peak_r)
            if found_hist_peak_l and found_hist_peak_r:
                left_lane = True
                right_lane = True
            elif found_hist_peak_r:
                right_lane = True
                left_lane = False
                hist_peak_l = hist_peak_l
                left_lane = False
            elif found_hist_peak_l:
                left_lane = True
                right_lane = False
                hist_peak_r = hist_peak_r
                right_lane = False
            else:
                hist_peak_r = hist_peak_r
                hist_peak_l = hist_peak_l
                right_lane = False
                left_lane = False

    hist_peak_r = hist_peak_r
    hist_peak_l = hist_peak_l
    # print("hist_peak_r:",hist_peak_r,"hist_peak_l", hist_peak_l,"left_lane", left_lane,"right_lane", right_lane,"inds", inds,count)
    return hist_peak_r, hist_peak_l, left_lane, right_lane, inds,count

left_lane_coeffs = 0
right_lane_coeffs = 0
def FindLanelines(hist_peak_l, hist_peak_r, left_lane, right_lane,inds):
    global left_lane_coeffs
    global right_lane_coeffs
    points_l,points_r = [], []
    line_width = 60
    for i, x in enumerate(inds[1]):
        # print("inds",inds[1])
        y = inds[0][i]
        # print(y)
        if int(hist_peak_l - line_width // 2) <= x <= int(hist_peak_l + line_width // 2):
            points_l.append([x, y])
        elif int(hist_peak_r - line_width // 2) <= x <= int(hist_peak_r + line_width // 2):
            points_r.append([x, y])

    points_l = np.asarray(points_l)
    points_r = np.asarray(points_r)

    # Find coefficients for the best fit lines for the points
    if not right_lane and not left_lane:
        pass
    elif (not left_lane) and len(points_r)>0:
        right_lane_coeffs = np.polyfit(points_r[:, 1], points_r[:, 0], 1)
    elif (not right_lane) and len(points_l)>0:
        left_lane_coeffs = np.polyfit(points_l[:, 1], points_l[:, 0], 1)
    else:
        if not(len(points_l))==0:
            left_lane_coeffs = np.polyfit(points_l[:, 1], points_l[:, 0], 1)
        if not(len(points_r))==0:
            right_lane_coeffs = np.polyfit(points_r[:,1], points_r[:,0], 1)
    return left_lane_coeffs, right_lane_coeffs


def CreateOverlay(left_lane_coeffs, right_lane_coeffs, frame):
    # Find the corners of the polygon that bounds the lane in the squared image
    x = [left_lane_coeffs[1], crop_height * left_lane_coeffs[0] + left_lane_coeffs[1],
         crop_height * right_lane_coeffs[0] + right_lane_coeffs[1], right_lane_coeffs[1]]
    y = [0, crop_height, crop_height, 0]

    line_thick = 10
    lane_image = np.zeros((crop_height, crop_width, 3), np.uint8)
    corners = []
    for i in range(4):
        corners.append((int(x[i]), int(y[i])))

    contour = np.array(corners, dtype=np.int32)
    cv2.drawContours(lane_image, [contour], -1, (203,192,255), -1)

    cv2.line(lane_image, corners[0], corners[1], (0, 255, 255),10)
    cv2.line(lane_image, corners[2], corners[3], (0, 255, 255), 10)




    # Find slope of midline
    p1 = (0, (left_lane_coeffs[1] + right_lane_coeffs[1]) // 2)
    p2 = (crop_height, (crop_height * left_lane_coeffs[0] + left_lane_coeffs[1] +
                  crop_height * right_lane_coeffs[0] + right_lane_coeffs[1]) // 2)
    lineslope = (p2[1] - p1[1]) / (p2[0] - p1[0])

    lane_image = lane_image
    H_inv = np.linalg.inv(H)
    warped_lane = Warp(H_inv, lane_image, frame.shape[0], frame.shape[1])

    x[0] -= line_thick / 2
    x[1] -= line_thick / 2
    x[2] += line_thick / 2
    x[3] += line_thick / 2

    X_s = np.array([x, y, np.ones_like(x)])
    sX_c = H.dot(X_s)
    X_c = sX_c / sX_c[-1]
    corners = []
    for i in range(4):
        corners.append((X_c[0][i], X_c[1][i]))

    # Overlay a green polygon that represents the lane
    contour = np.array(corners, dtype=np.int32)
    lane_overlay_img = frame.copy()
    cv2.drawContours(lane_overlay_img, [contour], -1, (0,0 ,0), -1)
    lane_overlay_img = cv2.bitwise_or(lane_overlay_img, warped_lane)

    overlay = cv2.addWeighted(frame, 0.5, lane_overlay_img, 0.5, 0)

    if -.05 <= lineslope <= .05:
        text = 'Going Straight'
    elif lineslope < -.05:
        text = 'Turning Right'
    else:
        text = 'Turning Left'
    overlay = cv2.putText(overlay, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                          1,(0, 0, 255), 2, cv2.LINE_AA)

    return lane_image, overlay

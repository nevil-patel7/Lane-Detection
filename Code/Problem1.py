import cv2
import numpy as np


# function to adjust the gamma value of the function
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


userdefined = int(input("Do you want to write to video-\nEnter '0' To Visualise:\n Enter '1' to write to video:"))
write_to_video = False
if userdefined == 0:
    write_to_video = False
else:
    write_to_video = True

# getting the video feed
camera = cv2.VideoCapture('Night Drive - 2689.mp4')
if write_to_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 29
    out = cv2.VideoWriter('Output_Problem1.mp4', fourcc, fps, (1920, 1080))

# checking if video is being played
while (camera.isOpened()):
    ret, frame = camera.read()
    # ret returns false at the end of video
    if ret:
        # blurring the image
        # blurred_img = cv2.GaussianBlur(frame,(7,7),0)

        # converting the image to HSV
        frame2hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_v = frame2hsv[:, :, 2]

        # finding the CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
        frame_clahe = clahe.apply(hsv_v)

        # setting the gamma value, increased values may cause noise
        gamma = 1.0
        frame_gamma = adjust_gamma(frame_clahe, gamma=gamma)

        # adding the last V layer back to the HSV image
        frame2hsv[:, :, 2] = frame_gamma

        # converting back from HSV to BGR format
        frame_improved = cv2.cvtColor(frame2hsv, cv2.COLOR_HSV2BGR)

        # showing the image
        cv2.imshow('improved_image', frame_improved)
        if write_to_video:
            # writing the video
            out.write(frame_improved)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # if ret is False release the video which will exit the loop
        camera.release()
        print("End Of Video")
# releasing the video feed
if write_to_video:
    out.release()
# camera.release()
cv2.destroyAllWindows()

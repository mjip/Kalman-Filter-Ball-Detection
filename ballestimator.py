#!/usr/bin/python3

import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import math

############################
# PARAMETERS TO VARY
############################
# Keep as default, specify as commandline args
bgr_color = 150, 80, 180
color_threshold = 80
e_iter = 1
d_iter = 5
min_radius = 10

###########################

hsv_color = cv2.cvtColor( np.uint8([[bgr_color]] ), cv2.COLOR_BGR2HSV)[0][0]
HSV_lower = np.array([hsv_color[0] - color_threshold, hsv_color[1] - color_threshold, hsv_color[2] - color_threshold])
HSV_upper = np.array([hsv_color[0] + color_threshold, hsv_color[1] + color_threshold, hsv_color[2] + color_threshold])


def detect_ball(frame, e_iter, d_iter, min_radius):
    x, y, radius = -1, -1, -1

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)
    mask = cv2.erode(mask, None, iterations=e_iter)
    mask = cv2.dilate(mask, None, iterations=d_iter)

    im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = (-1, -1)

    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(mask)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # check that the radius is larger than some threshold
        if radius > min_radius:
            #outline ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            #show ball center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    return center[0], center[1], radius

def apply_Kalman_filter_estimation(kf, x_i, y_i):
    # Assume the transition model/truth for the ball is a simple constant motion,
    # computed by taking the total distance travelled divided by the number of 
    # video frames
    # Assume the sensor model is the horizontal position component of the detected
    # blob as returned by the cv2 sensor.
    measured = np.array([[np.float32(x_i)], [np.float32(y_i)]])
    kf.correct(measured)
    predicted = kf.predict()
    return predicted

if __name__ == "__main__":

    try:
        filepath = sys.argv[1]
        e_iter = sys.argv[2]
        d_iter = sys.argv[3]
        min_radius = sys.argv[4]
    except:
        print("Usage: ./ballestimator.py /path/to/video erode_iterations dilate_iterations min_radius")
        print("Assuming video was specified, proceeding with default values: 1, 1, 20")

    try:
        cap = cv2.VideoCapture(filepath)
    except Exception as e:
        print("Error: video cannot be processed: {}".format(e))
        sys.exit(1)

    # Setup Kalman filter
    # cv2.KalmanFilter([dynamParams, measureParams[, controlParams[, type]]]) 
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    predicted_pos = np.zeros((2, 1), np.float32)

    # Lists used to plot path distance/position vs time
    x = []
    xa = []
    x0s = []
    t = []
    tt = 1 
    x0 = -1
    y0 = -1
    path_dist = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        prev_x = x0
        prev_y = y0

        try:
            x0, y0, r = detect_ball(frame, e_iter, d_iter, min_radius)
            predicted_pos = apply_Kalman_filter_estimation(kf, x0, y0)
        except:
            break
        print("Center at: ({0}, {1}), radius of {2}".format(x0, y0, r))
        print("Predicted pos: {0}{1}".format(predicted_pos[0],predicted_pos[1]))
        xa.append(predicted_pos[0])
        x0s.append(x0)
        t.append(tt/fps)
        tt += 1

        # Caculate path so far
        if prev_x != -1 and prev_y != -1:
            path_dist += math.sqrt(abs(prev_x - x0) + abs(prev_y - y0))
        x.append(path_dist)

        # Display the resulting frame
        # to a reasonable size based on my laptop dimensions
        cv2.line(img=frame, pt1=(predicted_pos[0], 0), pt2=(predicted_pos[0], int(height)), color=(255, 0, 255), thickness=5, lineType=8, shift=0)
        frame_resized = cv2.resize(frame, (1300, 730))
        cv2.imshow('frame', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    for p in range(len(x)):
        x[p] = float(x[p] / path_dist)

    # Absolute path dist
    plt.xlabel('Time (s)')
    plt.ylabel('X Position as Estimated by Kalman Filter (Absolute Pixels)')
    plt.plot(t, xa)
    plt.plot(t, x0s)
    plt.show()

    # Fractional path dist
    #plt.xlabel('Time (s)')
    #plt.ylabel('X Position (Fraction of Total Path Length)')
    #plt.plot(t, x)
    #plt.show()

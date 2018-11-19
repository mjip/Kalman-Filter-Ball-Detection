# Kalman-Filter-Ball-Detection
🔴🔴🔴 A Kalman filter estimation for blob detection in mp4 files.

Uses Python3 opencv2 to capture the video frame by frame, and contour detection to determine the position of the ball as it's 
moving across the screen. The BGR filter/colour threshold and minimum radius of a blob can be specified to narrow down the 
contours generated per frame. The eroding/dilating iterations (used on the frame to help with contour precision) can also be 
specified as either command line arguments or config options in the top of the file. 

## Usage
First install opencv2 for Python3, then run:
```python3 ballestimator.py /path/to/video/file [erode_iterations] [dilate_iterations] [min_radius]```
The video file must be specified, but the other arguments are optional. The defaults for the optional arguments are
1, 5, and 10 respectively. There are further configuration options in the header of ballestimator.py (the colour filter 
BGR value and threshold, for detecting contours in the image).

The video width/height can be adjusted by changing the imresize parameters (default 1300x730).

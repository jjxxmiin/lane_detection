# -*- coding: utf-8 -*- 

import numpy as np
import cv2
import os
import glob
import math

def convert_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def channel(img, ch):
    return img[:, :, ch]

def convert_lab(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]

def convert_hsv(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]

def convert_yuv(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]
	
def mask_white_yellow(image):
    converted = convert_hls(image)
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    whiteYellowImage = cv2.bitwise_and(image, image, mask = mask)
    return whiteYellowImage

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    scaled_sobel = None
    
    # Sobel x
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
    # Sobel y
    else:
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
        abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))

    # Threshold x gradient
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    magnitude = np.sqrt(np.square(sobelx)+np.square(sobely))
    abs_magnitude = np.absolute(magnitude)
    scaled_magnitude = np.uint8(255*abs_magnitude/np.max(abs_magnitude))
    mag_binary = np.zeros_like(scaled_magnitude)
    mag_binary[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude <= mag_thresh[1])] = 1
    
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    arctan = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(arctan)
    dir_binary[(arctan >= thresh[0]) & (arctan <= thresh[1])] = 1
    
    return dir_binary
	
def combined_s_gradient_thresholds(img,debug=False):

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(20, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.4))
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Threshold color channel
    s_thresh_min = 100
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(combined)
    
    combined_binary[(s_binary == 1) | (combined == 1)] = 1
    
    if debug == False:
        return combined, combined_binary
    else:
        return combined, combined_binary, s_binary, mag_binary, dir_binary

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
            
    cv2.fillPoly(mask, vertices, ignore_mask_color)
        
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)

def draw_lines(img, line, color=[0, 0, 255], thickness=10): # 선 그리기
    for x1,y1,x2,y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def average_lane(lane_data):
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    for data in lane_data:
        x1s.append(data[0][0])
        y1s.append(data[0][1])
        x2s.append(data[0][2])
        y2s.append(data[0][3])
    
    if np.isnan(np.mean(x1s)):
        x1 = np.nan_to_num(np.mean(x1s))
    else:
        x1 = np.mean(x1s)
    if np.isnan(np.mean(x2s)):
        x2 = np.nan_to_num(np.mean(x2s))
    else:
        x2 = np.mean(x2s)
    if np.isnan(np.mean(y1s)):
        y1 = np.nan_to_num(np.mean(y1s))
    else:
        y1 = np.mean(y1s)
    if np.isnan(np.mean(y2s)):
        y2 = np.nan_to_num(np.mean(y2s))
    else:
        y2 = np.mean(y2s)
    
    return int(x1), int(y1), int(x2), int(y2)

def perspective_transform(img):
    (h, w) = (img.shape[0], img.shape[1])
    source = np.float32([[w // 2 - 76, h * .525], [w // 2 + 76, h * .525], [-100, h], [w + 100, h]])
    #destination = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
    destination = np.float32([[200, 0], [w - 200, 0], [200, h], [w - 200, h]])
    m = cv2.getPerspectiveTransform(source, destination)
    m_inv = cv2.getPerspectiveTransform(destination, source)
    
    warped = cv2.warpPerspective(img, m, (w, h))
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
    
    return warped, unwarped, m, m_inv

def find_equation(line, lr=None, slope=False, y_intercept=False):      # 방정식 구하는 함수, lr은 'left'혹은 'right'
    for a,b,c,d in line:
        x1, y1, x2, y2 = a,b,c,d
        m = (float(y2-y1)/float(x2-x1))    
        k = y1 - m*x1                  # 절편
        if lr=='left':                 # 왼쪽 선일 때  
            result_x = 600 
        elif lr == 'right':            # 오른쪽 선일 때
            result_x = 100
        result_y = m*result_x + k        
    if slope==True and y_intercept==True:
        return m, k
    
    return int(result_x), int(result_y)

def find_intersection_pt(l_line, r_line):                                   # 교점 구하는 함수
    m1, b1 = find_equation(l_line, lr='left', slope=True, y_intercept=True)
    m2, b2 = find_equation(r_line, lr='right', slope=True, y_intercept=True)
    intersection_pt_x, intersection_pt_y = (float(b2-b1)/float(m1-m2)), m1*(b2-b1)/(m1-m2) +b1
    return int(intersection_pt_x), 350
	
	
def calibrate_camera(calib_images_dir, verbose=False):
    assert os.path.exists(calib_images_dir), '"{}" must exist and contain calibration images.'.format(calib_images_dir)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(os.path.join(calib_images_dir, 'calibration*.jpg'))
    # Step through the list and search for chessboard corners
    for filename in images:

        img = cv2.imread(filename)
		
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        pattern_found, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if pattern_found is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if verbose:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, pattern_found)
                cv2.imshow('img',img)
                cv2.waitKey(500)

    if verbose:
        cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


def undistort(frame, mtx, dist):
    frame_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx)

    return frame_undistorted

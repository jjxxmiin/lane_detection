# -*- coding: utf-8 -*- 

import cv2
import numpy as np
from imutils.video import WebcamVideoStream
from detection import *
import matplotlib.pyplot as plt

def main():
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='Example/camera_cal')
    
    capture = cv2.VideoCapture("./test/outside_clockwise.avi")
    img_w = 720 #img.shape[0]
    img_h = 380 #img.shape[1]
    
    # ROI
    pts = np.float32([[-100, img_h],[img_w // 2 - 76, img_h * .525], [img_w // 2 + 76,img_h * .525],  [img_w + 100, img_h]])
    while True:    
        ret, img = capture.read()

        img = cv2.resize(img,(img_w,img_h))
		
        # distort_img
        #distort_img = undistort(img, mtx, dist)
        
		# perspective
        #warped, unwarped, m, m_inv = perspective_transform(img)
        
        # binary
        combined, combined_binary = combined_s_gradient_thresholds(img)
        #combined, combined_binary, s_binary, mag_binary, dir_binary = combined_s_gradient_thresholds(img,debug=True)
        
        # mask -> gray -> blur -> canny
        making = mask_white_yellow(img)
        gray = grayscale(making)
        blur = gaussian_blur(gray, 5)
        cny = canny(blur,40,80)
        
        # 관심영역 추출
        vertices = np.array(pts, np.int32)
        roi = region_of_interest(cny, [vertices])
        
        #houghLines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
        hough = cv2.HoughLinesP(roi, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)

        if hough is not None:
            line_arr = np.squeeze(hough,axis=1)

            slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi

            # 수평 기울기 제한
            line_arr = line_arr[np.abs(slope_degree)<160]
            slope_degree = slope_degree[np.abs(slope_degree)<160]
            # 수직 기울기 제한
            line_arr = line_arr[np.abs(slope_degree)>95]
            slope_degree = slope_degree[np.abs(slope_degree)>95]
            # 필터링된 직선 버리기
            L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
            temp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            L_lines, R_lines = L_lines[:,None], R_lines[:,None]

            foundLinesImage = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            L_lines_exist = (L_lines.shape != (0L, 1L, 4L))
            R_lines_exist = (R_lines.shape != (0L, 1L, 4L))
            
            if L_lines_exist:
                L_line = np.array([average_lane(L_lines)])
                cv2.line(foundLinesImage, (L_line[0][0], L_line[0][1]), find_equation(L_line, 'left'), [0,0,255], 10)
            if R_lines_exist:
                R_line = np.array([average_lane(R_lines)])
                cv2.line(foundLinesImage, (R_line[0][2], R_line[0][3]), find_equation(R_line,'right'), [0,0,255], 10)
            if L_lines_exist and R_lines_exist:
                circle_x, circle_y=find_intersection_pt(L_line, R_line)
                cv2.circle(foundLinesImage, (circle_x,circle_y), 10, (0, 255, 255), -1)

            origWithFoundLanes = weighted_img(foundLinesImage,img)

        else:
            origWithFoundLanes = img

        cv2.imshow('image',origWithFoundLanes)
        cv2.imshow('combined',combined)
        cv2.imshow('combined_binary',combined_binary)
        #cv2.imshow('dis',distort_img)
        #cv2.imshow('warped',warped)
        #cv2.imshow('unwarped',unwarped)
        #cv2.imshow('dis',distort_img)
        #cv2.imshow('cny',cny)
        
        if cv2.waitKey(33) > 0: break

    capture.release()
    #capture.stop()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
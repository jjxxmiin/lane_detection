import cv2
import numpy as np
import math

def convert_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

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

def main():
    rho = 2
    theta = np.pi/180
    threshold = 100
    min_line_length = 100
    max_line_gap = 100
    
    #img = cv2.imread('../test/test_img.jpg', cv2.IMREAD_COLOR)

    capture = cv2.VideoCapture("../test/outside_clockwise.avi")

    while True:    
        ret, img = capture.read()
        img_w = 720#img.shape[0]
        img_h = 380#img.shape[1]
        img = cv2.resize(img,(img_w,img_h))

        ori_img = img

        img = mask_white_yellow(img)
        img = grayscale(img)
        img = gaussian_blur(img, 5)
        img = canny(img,40,80)

        yTopMask = img_h*0.55

        vertices = np.array([[0, img_h], [0, img_h*0.75], [img_w, img_h*0.75], [img_w,img_h]], np.int32)

        img = region_of_interest(img, [vertices])

        #houghLines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
        hough = cv2.HoughLinesP(img, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)

        foundLinesImage = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        for line in hough:
            for x1,y1,x2,y2 in line:
                cv2.line(foundLinesImage, (x1, y1), (x2, y2), [255, 0, 0], 7)

        origWithFoundLanes = weighted_img(foundLinesImage,ori_img)

        cv2.imshow('image',origWithFoundLanes)

        if cv2.waitKey(33) > 0: break

    capture.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
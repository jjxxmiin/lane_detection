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

def perspective_transform(img,src,dst):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
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
    

def main():
    capture = cv2.VideoCapture("../test/outside_clockwise.avi")

    img_w = 720 #img.shape[0]
    img_h = 380 #img.shape[1]
    
    # perspective transform
    src = np.float32([[img_w*0.2, img_h*0.5],[0, img_h],[img_w, img_h*0.5], [img_w,img_h]])
    dst = np.float32([[0, img_h*0.5],[0, img_h],[img_w, img_h*0.5], [img_w,img_h]])
    
    # ROI
    pts = np.array([[0, img_h], [img_w*0.2, img_h*0.6], [img_w*0.8, img_h*0.6], [img_w,img_h]])
    
    while True:    
        ret, img = capture.read()
        
        img = cv2.resize(img,(img_w,img_h))

        ori_img = img

        # mask -> gray -> blur -> canny
        img = mask_white_yellow(img)
        img = grayscale(img)
        img = gaussian_blur(img, 5)
        img = canny(img,40,80)
           
        # image perspective transform
        warped, unwarped, m, m_inv = perspective_transform(ori_img,src,dst)
        
        # 관심영역 추출
        vertices = np.array(pts, np.int32)
        img = region_of_interest(img, [vertices])
        
        # hough Line detection
        #houghLines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
        hough = cv2.HoughLinesP(img, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)

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
            #foundLinesImage = np.zeros((img_h, img_w), dtype=np.uint8)

            if L_lines is not None:
            L_line = np.array([average_lane(L_lines)])
            if np.sum(L_line) != 0:
                cv2.line(foundLinesImage, (L_line[0][0], L_line[0][1]), find_equation(L_line, 'left'), [0,0,255], 10)
            if R_lines is not None:
                R_line = np.array([average_lane(R_lines)])
                if np.sum(R_line) != 0:
                    cv2.line(foundLinesImage, (R_line[0][2], R_line[0][3]), find_equation(R_line, 'right'), [0,0,255], 10)
            if L_lines is not None and R_lines is not None:
                if np.sum(L_line) != 0 and np.sum(R_line) != 0:
                    circle_x, circle_y=find_intersection_pt(L_line, R_line)
    #             원: cv2.circle( , 중심, 반지름, 색, 두꼐)
                    cv2.circle(foundLinesImage, (circle_x,circle_y), 10, (0, 255, 255), -1)

            origWithFoundLanes = weighted_img(foundLinesImage,ori_img)

        else:
            origWithFoundLanes = ori_img

        cv2.imshow('image',origWithFoundLanes)
        cv2.imshow('warp',warped)
        cv2.imshow('unwarp',unwarped)

        if cv2.waitKey(33) > 0: break

    capture.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
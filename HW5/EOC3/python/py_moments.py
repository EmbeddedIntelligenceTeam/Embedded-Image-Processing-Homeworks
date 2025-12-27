
import cv2

def MOMENTS_Run(img):
        moments = cv2.moments(img, True) 
        huMoments = cv2.HuMoments(moments).reshape(7)
        return moments, huMoments

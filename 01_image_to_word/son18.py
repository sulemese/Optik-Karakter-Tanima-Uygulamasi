import cv2
import numpy as np
def detect_horizontal_lines(image):
    # Yatay kenarları algılama
    edges = cv2.Canny(image, 50, 150)
    horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=200, maxLineGap=20)
    
    return horizontal_lines

# Örnek kullanım
image_path = "C:/son.jpg"
image = cv2.imread(image_path)
horizontal_lines = detect_horizontal_lines(image)
cv2.imshow(horizontal_lines)

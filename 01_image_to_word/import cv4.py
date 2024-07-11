import cv2
import numpy as np

def sharpen_image(image, max_height=800):
    # Keskinleştirme kerneli oluştur
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    
    # Kernel ile görüntüyü filtreleme
    sharpened_image = cv2.filter2D(image, -1, kernel)
    
    # Eğer görüntü yüksekliği belirlenen maksimum yükseklikten büyükse yeniden boyutlandır
    if sharpened_image.shape[0] > max_height:
        scale_factor = max_height / sharpened_image.shape[0]
        sharpened_image = cv2.resize(sharpened_image, None, fx=scale_factor, fy=scale_factor)
    
    return sharpened_image

# Örnek kullanım
image_path = "C:/me.jpg"
image = cv2.imread(image_path)

sharpened_image = sharpen_image(image)

# Sonucu gösterme
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

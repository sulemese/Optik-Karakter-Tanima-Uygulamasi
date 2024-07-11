import cv2
import os

def binarize_image(image_path, max_height=800):
    # Görüntüyü yükle
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Görüntü yüklenemedi.")
        return
    
    # Eğer görüntü yüksekliği belirlenen maksimum yükseklikten büyükse yeniden boyutlandır
    if image.shape[0] > max_height:
        scale_factor = max_height / image.shape[0]
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    
    # Binarizasyon (Otsu'nun ikili form eşikleme yöntemi)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return binary

image_path = "C:/me.jpg"
binary = binarize_image(image_path)

# Görüntüyü ekrana uygun şekilde göster
cv2.imshow('Image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import os
import numpy as np


def word_segmentation(image_path):
    # Görüntüyü yükle
    image2 = cv2.imread(image_path)
    img_filt = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])

    # Görüntüyü keskinleştirme filtresini uygula
    image3 = cv2.filter2D(image2, -1, img_filt)
    
    # Görüntüyü ölçeklendir
    image = cv2.resize(image3, None, fx=0.5, fy=0.5)  # Yarı boyutuna ölçeklendir
    if image is None:
        print("Görüntü yüklenemedi.")
        return
    
    # Gri tonlama ve ikili formata dönüştürme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Genişletme ve erozyon
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Konturları bulma
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tüm konturların yüksekliklerini ölç
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    max_height = max(heights)

    # Her bir kontur için sınırlayıcı kutuları çizme ve kırpma
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Eşit yüksekliği kullanarak kutuyu çizme
        cv2.rectangle(image, (x-2, y), (x + w+2, y + max_height), (0, 255, 0), 2)
        
    # Sonucu gösterme
    cv2.imshow('Word Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Örnek kullanım
image_path = "C:/pictures/8.jpg"
word_segmentation(image_path)

import cv2
import numpy as np

def merge_horizontal(contours, threshold):
    merged_contours = []
    merged = contours[0]

    for current in contours:
        x, y, w, h = cv2.boundingRect(current)
        x2, y2, w2, h2 = cv2.boundingRect(merged)

        if y + h >= y2 and y <= y2 + h2 and x <= x2 + w2 + threshold:
            merged = np.concatenate((merged, current))
        else:
            merged_contours.append(merged)
            merged = current

    merged_contours.append(merged)
    return merged_contours
def word_segmentation(image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    
    # Görüntüyü keskinleştirme filtresini uygula (isteğe bağlı)
    img_filt = np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
    image = cv2.filter2D(image, -1, img_filt)
    
    # Görüntüyü ölçeklendir
    image = cv2.resize(image, None, fx=0.5, fy=0.5)  # Yarı boyutuna ölçeklendir
    if image is None:
        print("Görüntü yüklenemedi.")
        return
    
    # Gri tonlama ve ikili formata dönüştürme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Genişletme ve erozyon
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Konturları bulma
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Yatay birleştirme için eşik değeri
    merge_threshold = 10
    
    # Sınırlayıcı kutuları yatayda birleştirme
    merged_contours = merge_horizontal(contours, merge_threshold)
    
    # Büyük sınırlayıcı kutuları çizme
    for merged_contour in merged_contours:
        x, y, w, h = cv2.boundingRect(merged_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Sonucu gösterme
    cv2.imshow('Word Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Örnek kullanım
image_path = "C:/pictures/8.jpg"
word_segmentation(image_path)

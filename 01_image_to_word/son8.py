import cv2
import numpy as np

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
    
    # Konturları y koordinatlarına göre sırala
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    # Y koordinatları arasındaki fark eşiği
    line_threshold = 30
    
    # Satırları bulma
    lines = []
    current_line = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if not current_line:
            # İlk kontur, bir satır oluşturur
            current_line.append(contour)
        else:
            # Konturun y koordinatını kontrol et
            prev_x, prev_y, _, _ = cv2.boundingRect(current_line[-1])
            if y - prev_y <= line_threshold:
                # Aynı satırda, mevcut satırı güncelle
                current_line.append(contour)
            else:
                # Yeni bir satır başlat
                lines.append(current_line)
                current_line = [contour]
    
    # Son satırı ekle
    if current_line:
        lines.append(current_line)
    
    # Satırları çizme
    for line in lines:
        for contour in line:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Sonucu gösterme
    cv2.imshow('Word Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Örnek kullanım
image_path = "C:/pictures/10.jpg"
word_segmentation(image_path)

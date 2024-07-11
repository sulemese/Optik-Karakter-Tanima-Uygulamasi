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
    image = cv2.filter2D(image2, -1, img_filt)


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

    # Kelime koordinatları ve sınırlayıcı kutuları için listeler oluştur
    word_coordinates = {}
    word_boxes = []

    # Her bir kontur için sınırlayıcı kutuları çizme ve kırpma
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Sınırlayıcı kutuyu çizme
        cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2)

        # Kelime koordinatlarını ve sınırlayıcı kutularını listeye ekle
        word_coordinates[i] = (x, y)
        word_boxes.append((x, y, x + w, y + h))

    # Satırları belirlemek için kelimeleri sınırlayıcı kutularına göre düzenle
    word_boxes.sort(key=lambda box: box[1])  # Y koordinatlarına göre sırala

    # Her bir satır için sınırlayıcı kutu koordinatlarını hesapla ve çiz
    current_row = []
    for box in word_boxes:
        x, y, x2, y2 = box
        if not current_row or abs(current_row[0][1] - y) < 20:  # Satır boş veya y koordinatları yakınsa
            current_row.append(box)
        else:
            # Satırı işle
            min_x = min(coord[0] for coord in current_row)
            max_x = max(coord[2] for coord in current_row)
            min_y = min(coord[1] for coord in current_row)
            max_y = max(coord[3] for coord in current_row)
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
            current_row = [box]

    # Sonuçları göster
    cv2.imshow('Word Segmentation with Rows', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Örnek kullanım
image_path = "C:/me.jpg"
word_segmentation(image_path)

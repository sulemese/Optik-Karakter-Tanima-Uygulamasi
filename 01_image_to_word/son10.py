import cv2
import numpy as np


def word_segmentation(image, word_bboxes):
    # Görüntüyü gri tonlama ve ikili formata dönüştürme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Satırları belirleme
    lines = []
    current_line = []
    for x, y, w, h in word_bboxes:
        current_line.append((x, y, w, h))

        # Yeni bir satıra geçme kontrolü
        if y + h < image.shape[0] - 20:
            next_y = image.shape[0] - 20
            for x1, y1, w1, h1 in current_line:
                if y1 + h1 > next_y:
                    lines.append(current_line)
                    current_line = []
                    break

    # Her bir satır için sınırlayıcı kutuları çizme
    for line in lines:
        x_min = min(x for x, _, _, _ in line)
        y_min = min(y for _, y, _, _ in line)
        x_max = max(x + w for x, _, w, _ in line)
        y_max = max(y + h for _, y, _, h in line)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Sonucu gösterme
    cv2.imshow('Line Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Örnek kullanım
image = cv2.imread("C:/pictures/image2.jpg")

# Kelimelerin sınırlayıcı kutu koordinatlarını kodunuzdan alın
word_bboxes = [
    (100, 100, 50, 30),
    (150, 100, 50, 30),
    (200, 100, 50, 30),
    (100, 150, 50, 30),
    (150, 150, 50, 30),
    (200, 150, 50, 30),
]

word_segmentation(image, word_bboxes)

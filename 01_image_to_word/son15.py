import cv2
import numpy as np

def sort_contours(contours):
    # Sınırlayıcı kutuları x koordinatına göre sırala
    sorted_contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])
    return sorted_contours

def draw_contours(image, contours):
    # Sıralanmış konturları çiz
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
def main(image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize et
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Konturları bul
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Konturları yatayda sırala
    sorted_contours = sort_contours(contours)
    
    # Sıralanmış konturları çiz ve indeksleri yaz
    draw_contours(image, sorted_contours)
    
    # Sonucu göster
    cv2.imshow('Sorted Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Örnek kullanım
image_path = "C:/pictures/10.jpg"
main(image_path)

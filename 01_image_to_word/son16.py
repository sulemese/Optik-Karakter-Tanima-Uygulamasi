import cv2
import numpy as np

def draw_index(image, index, x, y, w, h):
    # İndexi sınırlayıcı kutunun içine kırmızı renkte yaz
    cv2.putText(image, str(index), (x+int(w/2)-5, y+int(h/2)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def sort_contours(contours):
    # Sınırlayıcı kutuları y koordinatına göre sırala
    sorted_contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])
    return sorted_contours

def sort_words_in_lines(contours):
    lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Her bir satır için minimum ve maksimum y koordinatlarını belirle
        found_line = False
        for line in lines:
            if line['min_y'] <= y <= line['max_y']:
                line['words'].append(contour)
                found_line = True
                break
        if not found_line:
            lines.append({'min_y': y, 'max_y': y+h, 'words': [contour]})
    # Her bir satır için sıralama yap
    for line in lines:
        line['words'] = sorted(line['words'], key=lambda contour: cv2.boundingRect(contour)[0])
    return lines

def word_segmentation(image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    img_filt = np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])

    # Görüntüyü keskinleştirme filtresini uygula
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
    
    # Sıralama yapmak için satırları bul
    lines = sort_words_in_lines(contours)

    # Her bir satır için sınırlayıcı kutuları çizme ve kırpma
    index = 1
    for line in lines:
        for contour in line['words']:
            x, y, w, h = cv2.boundingRect(contour)
            # Sınırlayıcı kutuyu çizme
            cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2)
            
            # İndexi sınırlayıcı kutunun içine yazma
            draw_index(image, index, x, y, w, h)
            index += 1

    # Sonucu gösterme
    cv2.imshow('Word Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Örnek kullanım
image_path = "C:/son.jpg"
word_segmentation(image_path)

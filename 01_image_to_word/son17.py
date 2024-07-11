import cv2
import numpy as np

def line_segmentation(image_path, output_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    
    # Görüntüyü ölçeklendir
    image = cv2.resize(image, None, fx=0.7, fy=0.7)  # %70 oranında ölçeklendir
    
    if image is None:
        print("Görüntü yüklenemedi.")
        return
    
    # Görüntüyü keskinleştirme filtresini uygula
    img_filt = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])
    image = cv2.filter2D(image, -1, img_filt)
    
    # Gri tonlama ve ikili formata dönüştürme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Genişletme ve erozyon
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Yatay projeksiyon (satırları tespit etmek için)
    horizontal_proj = np.sum(eroded, axis=1)
    
    # Satırların başlangıç ve bitişlerini tespit et
    row_indices = np.where(horizontal_proj > 0)[0]
    rows = []
    start_idx = row_indices[0]
    for i in range(1, len(row_indices)):
        if row_indices[i] != row_indices[i - 1] + 1:
            rows.append((start_idx, row_indices[i - 1]))
            start_idx = row_indices[i]
    rows.append((start_idx, row_indices[-1]))
    
    # Satırları ana görüntüde çizme ve numaralandırma
    for i, (start_row, end_row) in enumerate(rows):
        cv2.rectangle(image, (0, start_row), (image.shape[1], end_row), (0, 0, 0), 2)
        cv2.putText(image, f'{i + 1}', (10, start_row + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Sonucu kaydetme
    cv2.imwrite(output_path, image)

    # Sonucu gösterme
    cv2.imshow('Line Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Örnek kullanım
input_image_path = "C:/a/23.jpg"
output_image_path = "C:/a/23_segmented.jpg"
line_segmentation(input_image_path, output_image_path)

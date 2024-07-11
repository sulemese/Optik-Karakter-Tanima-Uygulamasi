import cv2
import numpy as np

def word_segmentation(image_path,output_image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    
    
    # Görüntüyü ölçeklendir
    image = cv2.resize(image, None, fx=0.7, fy=0.7)  # %70 oranında ölçeklendir
    
     # Sonucu kaydetme
    cv2.imwrite("C:/a/23_resize.jpg", image)
    
    if image is None:
        print("Görüntü yüklenemedi.")
        return
    
    # Görüntüyü keskinleştirme filtresini uygula
    img_filt = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])
    image = cv2.filter2D(image, -1, img_filt)
    
     # Sonucu kaydetme
    cv2.imwrite("C:/a/23_filter.jpg", image)
    
    # Gri tonlama ve ikili formata dönüştürme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Sonucu kaydetme
    cv2.imwrite("C:/a/23_gray.jpg", image)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
     # Sonucu kaydetme
    cv2.imwrite("C:/a/23_binary.jpg", binary)
    cv2.imshow('Binary Image', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
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
    
    bounding_boxes = []

    for (start_row, end_row) in rows:
        line_image = eroded[start_row:end_row, :]
        
        # Dikey projeksiyon (kelimeleri tespit etmek için)
        vertical_proj = np.sum(line_image, axis=0)
        
        # Kelimelerin başlangıç ve bitişlerini tespit et
        col_indices = np.where(vertical_proj > 0)[0]
        cols = []
        start_idx = col_indices[0]
        for i in range(1, len(col_indices)):
            if col_indices[i] != col_indices[i - 1] + 1:
                cols.append((start_idx, col_indices[i - 1]))
                start_idx = col_indices[i]
        cols.append((start_idx, col_indices[-1]))
        
        for (start_col, end_col) in cols:
            x, y, w, h = start_col, start_row, end_col - start_col, end_row - start_row
            bounding_boxes.append((x, y, w, h))
    
    # Bounding box'ları sıralama (yukarıdan aşağıya, soldan sağa)
    bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[1], box[0]))
    
    # Sıralı bounding box'ları ana görüntüde çizme ve numaralandırma
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        cv2.rectangle(image, (x-2, y-2), (x + w+2, y + h+2), (0,0 , 255), 2)
        cv2.putText(image, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
     
     # Sonucu kaydetme
    cv2.imwrite(output_image_path, image)
    
    # Sonucu gösterme
    cv2.imshow('Word Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Örnek kullanım
image_path = "C:/a/23.jpg"
output_image_path = "C:/a/23_segmented.jpg"
word_segmentation(image_path,output_image_path)

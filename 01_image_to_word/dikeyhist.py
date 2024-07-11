import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def word_segmentation(image_path, output_image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)

    if image is None:
        print("Görüntü yüklenemedi.")
        return
    
    # Görüntüyü keskinleştirme filtresini uygula
    img_filt = np.array([[-1, -1, -1],
                         [-1,  10, -1],
                         [-1, -1, -1]])

    image2 = cv2.filter2D(image, -1, img_filt)
    
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Genişletme ve erozyon
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=8)
    eroded = cv2.erode(dilated, kernel, iterations=3)
 
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
    
    for row_num, (start_row, end_row) in enumerate(rows):
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
        
        for col_num, (start_col, end_col) in enumerate(cols):
            x, y, w, h = start_col, start_row, end_col - start_col, end_row - start_row
            bounding_boxes.append((x, y, w, h, row_num, col_num))  # Satır ve sütun numaralarını ekle
    
    # Bounding box'ları sıralama (yukarıdan aşağıya, soldan sağa)
    bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[4], box[1], box[0]))
    
    # Ana klasörü oluşturma
    cropped_dir = "C:/cropped"
    os.makedirs(cropped_dir, exist_ok=True)
    
    # Sıralı bounding box'ları ana görüntüde çizme ve numaralandırma
    for (x, y, w, h, row_num, col_num) in bounding_boxes:
        # Dosya adını satır ve kelime numaralarını içerecek şekilde oluşturma
        cropped_image_path = os.path.join(cropped_dir, f"word_{row_num+1}_{col_num+1}.jpg")
        
        # Her bir bounding box'ı kırpma ve kaydetme
        cropped_image = gray1[y-2:y+h+2, x-2:x+w+2]
        cv2.imwrite(cropped_image_path, cropped_image)
        
        # Ana görüntüde dikdörtgen çizme ve numaralandırma
        cv2.rectangle(image, (x-1, y-1), (x + w+1, y + h+1), (0, 0, 255), 2)
        cv2.putText(image, f"{row_num+1}_{col_num+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Sonucu kaydetme
    cv2.imwrite(output_image_path, image)
    
    # Dikey histogram grafiğini çizme
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(vertical_proj)), vertical_proj)
    plt.title('Vertical Projection Histogram')
    plt.xlabel('Column Index')
    plt.ylabel('Sum of Pixel Values')
    plt.show()

    # Sonucu gösterme
    cv2.imshow('Word Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Örnek kullanım
image_path = "C:/a/24.jpg"
output_image_path = "C:/a/23_segmented.jpg"
word_segmentation(image_path, output_image_path)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def word_segmentation(image_path, output_image_path):
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
    cv2.imwrite("C:/a/23_binary.jpg", binary)
    
    # Yatay histogramı hesapla
    horizontal_proj = np.sum(binary, axis=1)
    
    # Histogram grafiğini çizdir
    plt.plot(horizontal_proj, np.arange(len(horizontal_proj)))
    plt.gca().invert_yaxis()  # Y eksenini tersine çevir
    plt.xlabel('Piksel Yoğunluğu')
    
    plt.title('Yatay Histogram Projeksiyonu')
    plt.savefig("C:/a/23_horizontal_projection.png")  # Grafiği kaydet
    plt.show()  # Grafiği göster

# Örnek kullanım
image_path = "C:/a/23.jpg"
output_image_path = "C:/a/23_segmented.jpg"
word_segmentation(image_path, output_image_path)

import cv2

def preprocess_image(image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        print("Görüntü yüklenemedi.")
        return None
    
    # Gürültüyü azaltma (Gauss filtresi uygulama)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Kontrastı artırma (Histogram eşitleme uygulama)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    
    # Binarizasyon (Otsu'nun ikili form eşikleme yöntemini uygulama)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return binary

# Örnek kullanım
image_path = "C:/s2.png"
preprocessed_image = preprocess_image(image_path)

# Sonucu gösterme
cv2.imshow('Preprocessed Image', preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

def xAxisKernel(kernel_size):
    # Custom kernel to connect horizontal blobs
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    kernel[int((kernel_size-1)/2), :] = 1
    return kernel

def find_lines(image_path, kernel_size=30):
    # Load the image
    img = cv2.imread(image_path)
    img_filt = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])

    # Görüntüyü keskinleştirme filtresini uygula
    img2 = cv2.filter2D(img, -1, img_filt)
    
    #görüntüyü ölçeklendir
    image = cv2.resize(img2, None, fx=0.5, fy=0.5)  # Yarı boyutuna ölçeklendir
    
    if image is None:
        print("Image not found.")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Close the image using custom kernel
    kernel = xAxisKernel(kernel_size)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find external contours (lines)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around lines
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "C:/pictures/image.jpg"
find_lines(image_path)

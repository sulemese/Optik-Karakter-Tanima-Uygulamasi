import streamlit as st
import cv2
import os
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder

# Streamlit UI
def main():
    st.title('OCR Prediction with Image Upload')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Display the uploaded image
        image = cv2.imread(temp_image_path)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Function for word segmentation
        def word_segmentation(image_path, output_image_path):
            # Load image
            image = cv2.imread(image_path)

            if image is None:
                st.error("Image could not be loaded.")
                return
            
            # Apply sharpening filter to the image
            img_filt = np.array([[-1, -1, -1],
                                [-1,  10, -1],
                                [-1, -1, -1]])
            
            image_sharpened = cv2.filter2D(image, -1, img_filt)

            # Convert to grayscale
            gray = cv2.cvtColor(image_sharpened, cv2.COLOR_BGR2GRAY)

            # Apply binary inverse thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # Morphological operations (dilation and erosion)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(binary, kernel, iterations=4)
            eroded = cv2.erode(dilated, kernel, iterations=3)

            # Horizontal projection (to detect lines)
            horizontal_proj = np.sum(eroded, axis=1)
            
            # Detect start and end of lines
            row_indices = np.where(horizontal_proj > 0)[0]
            rows = []
            start_idx = row_indices[0]
            for i in range(1, len(row_indices)):
                if row_indices[i] != row_indices[i - 1] + 1:
                    rows.append((start_idx, row_indices[i - 1]))
                    start_idx = row_indices[i]
            rows.append((start_idx, row_indices[-1]))
            
            bounding_boxes = []

            # Iterate through each row
            for row_num, (start_row, end_row) in enumerate(rows):
                line_image = eroded[start_row:end_row, :]
                
                # Vertical projection (to detect words)
                vertical_proj = np.sum(line_image, axis=0)
                
                # Detect start and end of words
                col_indices = np.where(vertical_proj > 0)[0]
                cols = []
                start_idx = col_indices[0]
                for i in range(1, len(col_indices)):
                    if col_indices[i] != col_indices[i - 1] + 1:
                        cols.append((start_idx, col_indices[i - 1]))
                        start_idx = col_indices[i]
                cols.append((start_idx, col_indices[-1]))
                
                # Iterate through each word in the row
                for col_num, (start_col, end_col) in enumerate(cols):
                    x, y, w, h = start_col, start_row, end_col - start_col, end_row - start_row
                    bounding_boxes.append((x, y, w, h, row_num, col_num))  # Add row and column numbers
            
            # Sort bounding boxes (top to bottom, left to right)
            bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[4], box[1], box[0]))
            
            # Create main directory
            cropped_dir = "cropped"
            os.makedirs(cropped_dir, exist_ok=True)
            
            # Draw and number sorted bounding boxes on main image
            for (x, y, w, h, row_num, col_num) in bounding_boxes:
                # Create file name including row and word numbers
                cropped_image_path = os.path.join(cropped_dir, f"word_{row_num+1}_{col_num+1}.jpg")
                
                # Crop and save each bounding box
                cropped_image = image[y-2:y+h+2, x-2:x+w+2]
                cv2.imwrite(cropped_image_path, cropped_image)
                
                # Draw rectangle and number on main image
                cv2.rectangle(image, (x-1, y-1), (x + w+1, y + h+1), (0, 0, 255), 2)
                cv2.putText(image, f"{row_num+1}_{col_num+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Save and display result
            cv2.imwrite(output_image_path, image)
            st.image(image, caption='Segmented Image', use_column_width=True)

        # Perform word segmentation
        output_image_path = "output_segmented.jpg"
        word_segmentation(temp_image_path, output_image_path)

        st.success(f"Segmented image saved to {output_image_path}")

        # Model and prediction part
        st.subheader('OCR Prediction')

        # Load model and configurations
        class ImageToWordModel(OnnxInferenceModel):
            def __init__(self, char_list: str, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.char_list = char_list

            def predict(self, image: np.ndarray):
                image = cv2.resize(image, (128, 32))  # Adjust size according to your model
                image_pred = np.expand_dims(image, axis=0).astype(np.float32)
                preds = self.model.run(None, {self.input_name: image_pred})[0]
                text = ctc_decoder(preds, self.char_list)[0]
                return text

        # Define model path and character list
        model_path = "mltu-main/Models/1_image_to_word/202211270035/model.onnx"
        char_list = "abcdefghijklmnopqrstuvwxyz"

        # Initialize model
        model = ImageToWordModel(model_path=model_path, char_list=char_list)

        # Function to process each cropped word image
        def process_word_image(image_path):
            # Read image
            new_image = cv2.imread(image_path)

            if new_image is None:
                return f"Error loading image {image_path}"

            try:
                # Predict text using the model
                prediction_text = model.predict(new_image)
                return prediction_text

            except Exception as e:
                return f"Error occurred while predicting for {image_path}: {e}"

        # Process each cropped word image and display results
        cropped_dir = "cropped"
        predictions_file = "C:/cropped/predictions.txt"
        results = {}

        # Iterate through cropped images
        for image_file in os.listdir(cropped_dir):
            if image_file.startswith("word_") and image_file.endswith(".jpg"):
                # Extract row and column numbers from file name
                parts = image_file.split("_")
                if len(parts) == 3 and parts[2].endswith(".jpg"):
                    row_num = int(parts[1])
                    col_num = int(parts[2].split(".")[0])

                    image_path = os.path.join(cropped_dir, image_file)
                    prediction_text = process_word_image(image_path)

                    if row_num not in results:
                        results[row_num] = {}
                    results[row_num][col_num] = prediction_text

        # Write predictions to file
        with open(predictions_file, "w") as f:
            for row_num in sorted(results.keys()):
                row_text = []
                for col_num in sorted(results[row_num].keys()):
                    row_text.append(results[row_num][col_num])
                f.write(f"Row {row_num}: {' '.join(row_text)}\n")

        st.success(f"OCR predictions saved to {predictions_file}")

# Run the app
if __name__ == '__main__':
    main()

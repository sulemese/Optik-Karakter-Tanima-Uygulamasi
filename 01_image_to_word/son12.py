import cv2
import os
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

if __name__ == "__main__":
    # Klasör yolunu belirtin
    folder_path = "C:/cropped/"
    
    # Model ve diğer gerekli yapılandırmaların yüklenmesi
    configs = BaseModelConfigs.load("mltu-main/Models/1_image_to_word/202211270035/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    
    # Sonuçların yazılacağı dosya
    output_text_path = "C:/cropped/predictions.txt"
    
    # Sonuçları tutmak için bir sözlük
    results = {}
    
    # Klasördeki dosyaları sırayla işleme
    for image_file in os.listdir(folder_path):
        if image_file.startswith("word_") and image_file.endswith(".jpg"):
            # Dosya adından satır ve sütun numaralarını çıkar
            parts = image_file.split("_")
            if len(parts) == 3 and parts[2].endswith(".jpg"):
                row_num = int(parts[1])
                col_num = int(parts[2].split(".")[0])
                
                image_path = os.path.join(folder_path, image_file)
                
                # Fotoğrafı oku
                new_image = cv2.imread(image_path)
                
                if new_image is None:
                    print(f"Error loading image {image_file}")
                    continue
                
                # Tahmini yap
                try:
                    prediction_text = model.predict(new_image)
                    print(f"Prediction for {image_file}: {prediction_text}")
                    
                    # Sonuçları sözlüğe ekle
                    if row_num not in results:
                        results[row_num] = {}
                    results[row_num][col_num] = prediction_text

                except Exception as e:
                    print(f"Error occurred while predicting for {image_file}: {e}")

    # Sonuçları bir dosyaya yaz
    with open(output_text_path, "w") as f:
        for row_num in sorted(results.keys()):
            row_text = []
            for col_num in sorted(results[row_num].keys()):
                row_text.append(results[row_num][col_num])
            f.write(f"Row {row_num}: {' '.join(row_text)}\n")

    print(f"Predictions saved to {output_text_path}")

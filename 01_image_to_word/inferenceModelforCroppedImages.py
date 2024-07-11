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

    # Klasördeki tüm dosyaları al
    image_files = os.listdir(folder_path)

    for image_file in image_files:
        # Dosyanın tam yolunu oluştur
        image_path = os.path.join(folder_path, image_file)

        # Fotoğrafı oku
        new_image = cv2.imread(image_path)

        # Tahmini yap
        try:
            prediction_text = model.predict(new_image)
            print(f"Prediction for {image_file}: {prediction_text}")

            # Tahmini fotoğrafın başlığı olarak ayarla
            cv2.namedWindow("Image")
            cv2.setWindowTitle("Image", f"Prediction: {prediction_text}")

            # Fotoğrafı göster
            cv2.imshow("Image", new_image)

            # Kullanıcı enter tuşuna basana kadar bekle
            cv2.waitKey(0)

        except Exception as e:
            print(f"Error occurred while predicting for {image_file}: {e}")

    # Tüm işlem bittikten sonra tüm pencereleri kapat
    cv2.destroyAllWindows()

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
    # Belirli bir fotoğraf dosyası yolu
    new_image_path ="C:/12.png"

    # Modelin ve diğer gerekli yapılandırmaların yüklenmesi
    configs = BaseModelConfigs.load("mltu-main/Models/1_image_to_word/202211270035/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    # Fotoğrafın okunması
    new_image = cv2.imread(new_image_path)
    

    # Tahminin yapılması
    try:
        prediction_text = model.predict(new_image)
        print(f"Prediction for {os.path.basename(new_image_path)}: {prediction_text}")

        # Tahmini fotoğrafın başlığı olarak ayarlama
        cv2.namedWindow("Image")
        cv2.setWindowTitle("Image", f"Prediction: {prediction_text}")

        # Fotoğrafı ekranda gösterme
        cv2.imshow("Image", new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred while predicting for {os.path.basename(new_image_path)}: {e}")

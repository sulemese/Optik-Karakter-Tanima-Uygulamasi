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

    # Klasördeki toplam dosya sayısını al
    total_files = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])

    # Dosyaları sırayla işleme
    for i in range(1, total_files + 1):  # 1'den toplam dosya sayısına kadar numaralandırılmış dosyaları işlemek için
        image_file = f"word_{i}.jpg"
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

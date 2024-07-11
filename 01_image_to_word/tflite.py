import tensorflow as tf

# .onnx modelini TensorFlow modeline yükleyin
onnx_model = tf.saved_model.load('C:/mltu-main/mltu-main/Models/02_captcha_to_text/202212211205/model.onnx')

# TFLite dönüştürücüsünü oluşturun ve modeli dönüştürün
converter = tf.lite.TFLiteConverter.from_saved_model(onnx_model)
tflite_model = converter.convert()

# TFLite modelini dosyaya kaydedin
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model başarıyla TFLite formatına dönüştürüldü ve kaydedildi.")

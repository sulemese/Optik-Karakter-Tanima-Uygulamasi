import streamlit as st
from PIL import Image

from streamlit_cropper import st_cropper

def main():
    st.title("Görüntü Kırpma Uygulaması")

    # Görüntü yükleme arayüzü
    uploaded_file = st.file_uploader("Lütfen bir görüntü yükleyin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Yüklenen dosyanın bir görüntü olup olmadığını kontrol et
        if uploaded_file.type.startswith("image"):
            # Yüklenen dosyayı görüntü olarak aç
            img = Image.open(uploaded_file)

            # Görüntüyü görüntüle
            st.image(img, caption="Yüklenen Görüntü", use_column_width=True)

            # Görüntü kırpma işlevselliği
            cropped_image = st_cropper(img)

            if cropped_image is not None:
                # Kırpılmış görüntüyü görüntüle
                st.image(cropped_image, caption="Kırpılmış Görüntü", use_column_width=True)
        else:
            st.error("Lütfen bir görüntü dosyası yükleyin.")

if __name__ == "__main__":
    main()

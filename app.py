import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Grayscale, ToPILImage
import io
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from torchvision.utils import save_image


def transform_upscale(image, upscale):
    image = Image.open(image).convert("RGB")
    image = Compose(
        [
            # Grayscale(),
            Resize(image.size[1] * upscale),
            ToTensor(),
        ]
    )(image)

    return image


def transform_display(image):
    return ToPILImage()(image)


def zipFromImages(out_images, uploaded_images):
    images = []
    for i, image in enumerate(out_images):
        format = Path(uploaded_images[i].name).suffix[1:].upper()
        if format == "JPG":
            format = "JPEG"

        buffer = io.BytesIO()
        save_image(image, buffer, format=format)

        out_images[i] = buffer
        images.append(buffer)

    memory = BytesIO()
    with ZipFile(memory, "w") as zip_file:
        for i, bytes in enumerate(images):
            zip_file.writestr(f"{uploaded_images[i].name}", bytes.getvalue())

    return memory


@st.cache(max_entries=1)
def get_model():
    return torch.load("./savedModels/runetPERCEPTUAL/_model.pt")


def app():
    st.set_page_config(
        layout="wide",
        page_title="Comics upscaler",
        page_icon="icon_comic.png",
    )

    model = get_model()
    st.header("Upscale And Enhance Your Comics Images")
    st.write(
        "Easy way to enhance and upscale your old forgotten images of comics books. It uses pre-trained deep neural network created specifically for this task. __We do not store uploaded images__."
    )
    st.write(
        "__How to use this website?__ \n 1. Choose upscale factor \n 2. Upload images \n 3. Download zip file with upscaled images"
    )
    upscale = st.radio("Upscale factor", [2, 4])
    uploaded_images = st.file_uploader(
        "Upload mage here: Max size: 200x200", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if len(uploaded_images):
        out_images = []

        with st.spinner("Please wait..."):
            for file in uploaded_images:
                in_image = file
                out_image = transform_upscale(in_image, upscale)
                with torch.no_grad():
                    out_image = model(out_image.unsqueeze(0))
                out_image = out_image.squeeze(0)
                out_images.append(out_image)

        st.download_button("Download upscaled images", zipFromImages(out_images, uploaded_images), f"images.zip")

        st.write("Example:")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Input image")
            st.image(in_image, use_column_width="always")
        with col2:
            st.caption("Output image")
            st.image(out_images[-1], use_column_width="always")

    st.subheader("About project")
    st.write(
        "This project is created as a bachelor thesis. "
        "The theme of the thesis is Image Super-Resolution Using Deep Learning. "
        "The model used for this task was based on U-Net architecture. Name of used model is RUNET and is described in this "
        "[paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WiCV/Hu_RUNet_A_Robust_UNet_Architecture_for_Image_Super-Resolution_CVPRW_2019_paper.pdf). "
        "Nowdays methods of using deep neural networks are better then using basic interpolations method. "
        "I focues on comics images for their popularity and my relationship for them."
    )
    st.image("dataset-min.jpg")

    st.caption("© 2022 Author Peter Zdravecký")


if __name__ == "__main__":
    app()

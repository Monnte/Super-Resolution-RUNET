"""
Webapplication for super-resolution task using pretrained models.

:filename app.py
:date 12.02.2022
:author Peter Zdravecký
:email xzdrav00@stud.fit.vutbr.cz

TODO: Memory optimizations for model -> lower depth.
    : Do caching for model after error from streamlit is fixed. (Error when using torch script).
"""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor
from torchvision.utils import save_image


def prepare_image(image, upscale_factor):
    """Open image and upscale by scale factor."""
    image = Image.open(image).convert("RGB")

    return Compose(
        [
            Resize(
                (image.size[1] * upscale_factor, image.size[0] * upscale_factor),
                interpolation=InterpolationMode.BICUBIC,
            ),
            ToTensor(),
        ]
    )(image)


def zipFromImages(out_images, uploaded_images):
    """Make zip file from upscaled images."""
    images = []
    for i, image in enumerate(out_images):
        format = Path(uploaded_images[i].name).suffix[1:].upper()
        if format == "JPG":
            format = "JPEG"

        buffer = BytesIO()
        save_image(image, buffer, format=format)

        out_images[i] = buffer
        images.append(buffer)

    data = BytesIO()
    with ZipFile(data, "w") as zip_file:
        for i, bytes in enumerate(images):
            zip_file.writestr(f"{uploaded_images[i].name}", bytes.getvalue())

    return data


def get_model_2x():
    """Load model for upscale factor 2."""
    model = torch.jit.load("model_2x.pt")
    model.eval()
    return model


def get_model_4x():
    """Load model for upscale factor 4."""
    model = torch.jit.load("model_4x.pt")
    model.eval()
    return model


def app():
    """Main function for the application."""
    st.set_page_config(
        layout="wide",
        page_title="Comics upscaler",
        page_icon="icon.png",
    )

    st.header("Upscale And Enhance Your Comics Images")
    st.write(
        "Easy way to enhance and upscale your old forgotten images of comics books. It uses pre-trained deep neural network created specifically for this task. __We do not store uploaded images__."
    )
    st.write(
        "__How to use this website?__ \n 1. Choose upscale factor \n 2. Upload images \n 3. Download zip file with upscaled images"
    )
    upscale = st.radio("Upscale factor", [2, 4])

    if upscale == 2:
        model = get_model_2x()
    else:
        model = get_model_4x()

    uploaded_images = st.file_uploader("Upload image here:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # If images are uploaded do work
    if len(uploaded_images):
        out_images = []

        with st.spinner("Please wait..."):
            for file in uploaded_images:
                in_image = file
                out_image = prepare_image(in_image, upscale)
                with torch.no_grad():
                    out_image = model(out_image.unsqueeze(0))
                out_image = out_image.squeeze(0)
                out_images.append(out_image)

        st.download_button(
            "Download upscaled images",
            zipFromImages(out_images, uploaded_images),
            f"result.zip",
            on_click=uploaded_images.clear(),
        )

        st.write("Preview:")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Input image")
            st.image(in_image)
        with col2:
            st.caption("Output image")
            st.image(out_images[-1])

    st.subheader("About project")
    st.write("The application uses trained deep neural network for super-resolution task. Bachelor thesis project.")
    st.image("dataset.jpg")
    st.caption("© 2022 Author Peter Zdravecký")


if __name__ == "__main__":
    app()

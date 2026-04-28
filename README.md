# Visual Guessing Machine

A Streamlit app that lets a user upload an image and get image classification predictions from a pretrained torchvision model.

## What it does

- uploads an image file
- preprocesses it with the correct pretrained transforms
- runs it through a pretrained MobileNetV3 model
- shows the top predicted labels with confidence scores

## Model used

This app uses `mobilenet_v3_large` with pretrained `MobileNet_V3_Large_Weights.DEFAULT` from `torchvision.models`.

## How to run

1. create and activate a virtual environment
2. install the packages
3. run the app with Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py

import streamlit as st
import pandas as pd
from io import StringIO






st.write("## Generate Image Caption")
st.write(
    ":dog: Try uploading an image to generate a narrative for that image"
)
st.sidebar.write("## Upload and download :gear:")



MAX_FILE_SIZE = 5 * 1024 * 1024  
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        # fix_image(upload=my_upload)
        pass
else:
#     fix_image("./zebra.jpg")
    pass



from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
import requests
import torch
import numpy as np
from PIL import Image
import pickle

import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')



#for local machine running use this
model_raw = torch.load('/Users/gauravmahajan/Downloads/model_test_whole_2may.pth')
model_raw.eval()










image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer       = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

import requests
from PIL import Image
import matplotlib.pyplot as plt

def show_n_generate(file_path, greedy=True, model=model_raw):
    try:
        image = Image.open(file_path)
        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        col1.write("Original Image :camera:")
        col1.image(image)
        
   
        

        if greedy:
            generated_ids = model.generate(pixel_values, max_new_tokens=30)
        else:
            generated_ids = model.generate(
                pixel_values,
                do_sample=True,
                max_new_tokens=30,
                top_k=5)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        col2.write(generated_text)
        
          

    except Exception as e:
        st.error("Please upload an image first.")

# Example usage:
# file_path = ""
# show_n_generate(my_upload, greedy=False)
if st.button("Generate Caption"):
    # Call your function when the button is clicked
    show_n_generate(my_upload, greedy=False)



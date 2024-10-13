import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and vocabulary
model = load_model('image_caption_generator.h5')

# Load vocabulary
with open('updated_vocab.npy', 'rb') as f:
    # vocab = np.load(f, allow_pickle=True).item()
    vocab_array = np.load('updated_vocab.npy', allow_pickle=True)
    vocab = {word: idx for word, idx in vocab_array}



max_length = 30 

def preprocess_image(image):
    # Resize the image
    image = load_img(image, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def generate_caption(image):
    # Extract features from the image
    features = model.predict(image)
    
    # Generate the caption

    caption = []  # Initialize caption
    current_word_index = vocab['<start>']  # Start with the start token
    for _ in range(max_length):
        # Predict next word
        preds = model.predict([features, current_word_index])
        current_word_index = np.argmax(preds[0])  # Get index of the predicted word
        caption.append(current_word_index)  # Append the predicted index to the caption
        if current_word_index == vocab['<end>']:  # Break if end token is predicted
            break
    return ' '.join([vocab[i] for i in caption])  # Convert indices back to words

# Streamlit app
st.title("Image Captioning App")
st.write("Upload an image to generate a caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = preprocess_image(uploaded_file)
    
    # Generate a caption for the uploaded image
    if st.button("Generate Caption"):
        caption = generate_caption(image)
        st.write("Generated Caption:", caption)

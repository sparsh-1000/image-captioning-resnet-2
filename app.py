import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load models
resnet_model = load_model('resnet50_flickr8k.h5')
captioning_model = load_model('image_caption_generator.h5')

# load vocabulary
with open('updated_vocab.npy', 'rb') as f:
    vocab_array = np.load('updated_vocab.npy', allow_pickle=True)
    vocab = {word: idx for word, idx in vocab_array}

max_length = 34  

def preprocess_image(image):
    image = load_img(image, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def extract_features(image):
    features = resnet_model.predict(image)
    return features

def generate_caption(image):
    features = extract_features(image)
    caption = []
    current_word_index = vocab['<start>']
    for _ in range(max_length):
        sequence = pad_sequences([[current_word_index]], maxlen=max_length, padding='post')
        preds = captioning_model.predict([features, sequence], verbose=0)
        current_word_index = np.argmax(preds[0])
        caption.append(current_word_index)
        if current_word_index == vocab['<end>']:
            break
    return ' '.join([list(vocab.keys())[list(vocab.values()).index(i)] for i in caption])

# streamlit start
st.title("Image Captioning App")
st.write("Upload an image to generate a caption.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    
    if st.button("Generate Caption"):
        caption = generate_caption(image)
        st.write("Generated Caption:", caption)
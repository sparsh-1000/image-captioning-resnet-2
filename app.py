import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and vocabulary
model = load_model('image_caption_generator.h5')

# Load vocabulary
vocab_array = np.load('vocab.npy', allow_pickle=True).item()  # Load the updated vocab
vocab = {word: idx for word, idx in vocab_array.items()}  # Create a dictionary for quick access

# Add start and end tokens
start_token = '<start>'
end_token = '<end>'
vocab[start_token] = max(vocab.values()) + 1  # Assign a new ID for <start>
vocab[end_token] = max(vocab.values()) + 2  # Assign a new ID for <end>

# Reverse vocabulary for converting indices back to words
inverse_vocab = {idx: word for word, idx in vocab.items()}

max_length = 30  # Set max length for generated captions

def preprocess_image(uploaded_file):
    # Convert the uploaded file to an OpenCV image
    img = load_img(uploaded_file, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Reshape to fit model input
    img /= 255.0  # Normalize image
    return img

def generate_caption(image):
    # Extract features from the image
    features = model.predict(image).reshape(1, -1)  # Reshape for model input

    # Start with the start token
    caption = [vocab[start_token]]
    
    for _ in range(max_length):
        # Prepare the input for the model
        encoded = pad_sequences([caption], maxlen=max_length, padding='post')
        preds = model.predict([features, encoded])  # Get predictions
        current_word_index = np.argmax(preds[0])  # Get the index of the predicted word
        
        caption.append(current_word_index)  # Append predicted index

        if current_word_index == vocab[end_token]:  # Break if end token is predicted
            break
            
    # Convert indices back to words
    return ' '.join([inverse_vocab[i] for i in caption if i in inverse_vocab])  # Use the inverse vocabulary

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

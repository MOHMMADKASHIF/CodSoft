import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Step 1: Feature Extraction with Pre-trained ResNet

# Load ResNet50 model pre-trained on ImageNet
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    features = model.predict(img_data)
    return features

# Example image
img_path = 'example.jpg'
features = extract_features(img_path, resnet_model)
print(f"Extracted features shape: {features.shape}")

# Step 2: Define the Captioning Model

def define_captioning_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# Example parameters
vocab_size = 5000
max_length = 34
caption_model = define_captioning_model(vocab_size, max_length)
caption_model.summary()

# Step 3: Training the Model

# Example of synthetic training data (usually you'd use a dataset like MSCOCO)
# image_features: array of shape (num_samples, 2048)
# sequences: array of shape (num_samples, max_length)
# next_words: array of shape (num_samples, vocab_size)

# Dummy data for illustration
num_samples = 1000
image_features = np.random.rand(num_samples, 2048)
sequences = np.random.randint(1, vocab_size, size=(num_samples, max_length))
next_words = np.random.randint(0, vocab_size, size=(num_samples, vocab_size))

# Training the model
caption_model.fit([image_features, sequences], next_words, epochs=1, batch_size=64, verbose=2)

# Step 4: Generating Captions

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Assume we have a tokenizer fitted on the training captions
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(['a dataset of captions...'])

# Generate caption for the example image
photo_features = extract_features(img_path, resnet_model)
caption = generate_caption(caption_model, tokenizer, photo_features, max_length)
print(f"Generated caption: {caption}")
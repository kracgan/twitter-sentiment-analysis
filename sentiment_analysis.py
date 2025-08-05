# ==============================================================================
#  1. IMPORTS AND GLOBAL VARIABLES
# ==============================================================================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# --- Configuration Variables ---
DATASET_PATH = 'training.1600000.processed.noemoticon.csv' # IMPORTANT: Change this path
DATASET_COLS = ['sentiment', 'id', 'date', 'query', 'user', 'text']
DATASET_ENCODING = 'ISO-8859-1'
SAMPLE_SIZE = 200000  # Use a fraction of the data for faster training

# --- Model Hyperparameters ---
VOCAB_SIZE = 10000     # Max number of words in the vocabulary
MAX_LENGTH = 60      # Max length for a tweet
EMBEDDING_DIM = 128    # Dimension of the word embeddings
OOV_TOKEN = "<OOV>"    # Token for out-of-vocabulary words

# ==============================================================================
#  2. DATA LOADING AND PREPROCESSING
# ==============================================================================
print("Step 2: Loading and Preprocessing Data...")

# Load the dataset
df = pd.read_csv(DATASET_PATH, encoding=DATASET_ENCODING, names=DATASET_COLS)

# Take a smaller sample for faster processing
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)

# --- Preprocessing Function ---
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    return text

# Apply preprocessing
df_sample['cleaned_text'] = df_sample['text'].apply(preprocess_text)

# Prepare labels (0 for negative, 1 for positive)
df_sample['sentiment'] = df_sample['sentiment'].replace(4, 1)

print("Data loading and preprocessing complete.")
print(df_sample[['cleaned_text', 'sentiment']].head())

# ==============================================================================
#  3. TOKENIZATION AND PADDING
# ==============================================================================
print("\nStep 3: Tokenizing and Padding Sequences...")

# Split data into training and testing sets
X = df_sample['cleaned_text'].values
y = df_sample['sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

print(f"Shape of padded training data: {X_train_padded.shape}")
print(f"Shape of padded testing data: {X_test_padded.shape}")

# ==============================================================================
#  4. BUILDING THE LSTM MODEL
# ==============================================================================
print("\nStep 4: Building the LSTM Model...")

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    Bidirectional(LSTM(64, return_sequences=True)), # Bidirectional helps learn context from both directions
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# ==============================================================================
#  5. TRAINING THE MODEL
# ==============================================================================
print("\nStep 5: Training the Model...")
history = model.fit(X_train_padded, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_data=(X_test_padded, y_test),
                    verbose=1)

print("Model training complete.")

# ==============================================================================
#  6. EVALUATING THE MODEL
# ==============================================================================
print("\nStep 6: Evaluating the Model...")

# Plot accuracy and loss
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Final evaluation on the test set
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

# ==============================================================================
#  7. INTERACTIVE PREDICTION LOOP
# ==============================================================================
print("\nStep 7: Interactive Sentiment Analysis demo is Ready!")
print("Enter a sentence to analyze, or type 'quit' to exit.")
print("after exiting the demo, the model and tokenizer will be saved.")

def predict_sentiment(text):
    # Preprocess, tokenize, and pad the input text
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # Predict sentiment
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return f"Sentiment: {sentiment} (Score: {prediction:.4f})"

# --- Loop for continuous user input ---
while True:
    user_input = input("\nEnter a sentence: ")
    
    # Check if the user wants to exit
    if user_input.lower() == 'quit':
        print("Exiting the analyzer.")
        break
        
    # Ensure the input is not empty
    if not user_input.strip():
        print("Please enter a valid sentence.")
        continue

    # Get and print the prediction
    result = predict_sentiment(user_input)
    print(result)
# ==============================================================================
#  8. SAVING THE MODEL AND TOKENIZER
# ==============================================================================
print("\nStep 8: Saving the model and tokenizer...")

# Save the trained model
model.save('sentiment_lstm_model.keras')

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and tokenizer have been saved successfully.")
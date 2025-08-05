import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Loading trained model and tokenizer...")

# --- Load the saved model and tokenizer ---
try:
    model = tf.keras.models.load_model('sentiment_lstm_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error loading files: {e}")
    print("Please make sure 'sentiment_lstm_model.h5' and 'tokenizer.pickle' are in the same directory.")
    exit()

# --- Re-use the same constants and preprocessing function from training ---
MAX_LENGTH = 60      # This MUST be the same value used during training
OOV_TOKEN = "<OOV>"    # This MUST be the same value used during training

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return f"Sentiment: {sentiment} (Score: {prediction:.4f})"

# --- Interactive loop for predictions ---
print("\n✅ Model loaded successfully. Ready for predictions.")
print("Enter a sentence to analyze, or type 'quit' to exit.")

while True:
    user_input = input("\nEnter a sentence: ")
    if user_input.lower() == 'quit':
        print("Exiting. Goodbye! 👋")
        break
    if not user_input.strip():
        print("Please enter a valid sentence.")
        continue
    
    result = predict_sentiment(user_input)
    print(result)
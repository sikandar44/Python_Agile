import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Step 1: Read and parse messages from the text file
def read_messages_from_file(file_path):
    messages = []
    with open(file_path, 'r') as file:
        # Read the entire file content
        file_content = file.read()
        # Use regular expression to find all messages within curly braces
        message_matches = re.findall(r'\{(.*?)\}', file_content, re.DOTALL)
        # Append each message to the list
        for match in message_matches:
            messages.append(match.strip())
    return messages


# Step 2: Preprocess messages for input to the LSTM model
def preprocess_messages(messages):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(messages)
    sequences = tokenizer.texts_to_sequences(messages)
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return padded_sequences, tokenizer


# Step 3: Define and train the LSTM model
def train_lstm_model(padded_sequences, vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, input_length=padded_sequences.shape[1]),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(padded_sequences, padded_sequences, epochs=10)
    return model


# Step 4: Translate new messages using the trained LSTM model
def translate_messages(messages, model, tokenizer):
    sequences = tokenizer.texts_to_sequences(messages)
    padded_sequences = pad_sequences(sequences, maxlen=model.input_shape[1], padding='post')
    translated_sequences = model.predict_classes(padded_sequences)
    translated_messages = tokenizer.sequences_to_texts(translated_sequences)
    return translated_messages


# Main function
def main():
    # Step 1: Read and parse messages from the text file
    file_path = 'messages.txt'
    messages = read_messages_from_file(file_path)

    # Step 2: Preprocess messages for input to the LSTM model
    padded_sequences, tokenizer = preprocess_messages(messages)

    # Step 3: Train the LSTM model
    vocab_size = len(tokenizer.word_index) + 1
    lstm_model = train_lstm_model(padded_sequences, vocab_size)

    # Step 4: Translate new messages using the trained LSTM model
    translated_messages = translate_messages(messages, lstm_model, tokenizer)
    for original, translated in zip(messages, translated_messages):
        print(f"Original: {original}\nTranslated: {translated}\n")


if __name__ == "__main__":
    main()

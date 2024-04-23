import os
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Step 1: Parse the text file and extract the messages
def extract_messages_from_folder(folder_path):
    messages = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            content = file.read()
            messages.extend(re.findall(r'@internalPortMessageWithAspsSip.*?}', content, re.DOTALL))
    return messages

# Step 2: Preprocess the messages
def preprocess_messages(messages):
    processed_messages = []
    for message in messages:
        processed_message = message.replace('\n', '').replace('\t', '')
        processed_messages.append(processed_message)
    return processed_messages

# Step 3: Build the LSTM model
def build_model(vocabulary_size, max_sequence_length):
    model = Sequential()
    model.add(Embedding(vocabulary_size, 128, input_length=max_sequence_length))
    model.add(LSTM(128))
    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 4: Tokenize and pad sequences
def tokenize_and_pad_sequences(messages):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(messages)
    sequences = tokenizer.texts_to_sequences(messages)
    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return padded_sequences, tokenizer, max_sequence_length

# Step 5: Train the LSTM model
def train_model(model, padded_sequences):
    # Assuming you have labels for training (not provided in the code)
    labels = np.random.randint(2, size=(len(padded_sequences),))  # Example random labels
    model.fit(padded_sequences, labels, epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed
    return model

# Step 6: Translate the messages using the LSTM model
def translate_messages_to_folder(messages, model, tokenizer, max_sequence_length, output_folder):
    translated_messages = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, message in enumerate(messages):
        sequence = tokenizer.texts_to_sequences([message])
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
        translated_sequence = model.predict(padded_sequence)
        translated_message = tokenizer.sequences_to_texts([np.argmax(translated_sequence)])
        translated_messages.append(translated_message[0])
        # Write translated message to output file
        output_file_path = os.path.join(output_folder, f'translated_message_{i}.txt')
        with open(output_file_path, 'w') as output_file:
            output_file.write(translated_message[0])
    return translated_messages

# Example usage
input_folder = 'input'
output_folder = 'output'

# Extract messages from input folder
messages = extract_messages_from_folder(input_folder)
processed_messages = preprocess_messages(messages)
padded_sequences, tokenizer, max_sequence_length = tokenize_and_pad_sequences(processed_messages)
model = build_model(len(tokenizer.word_index) + 1, max_sequence_length)
model = train_model(model, padded_sequences)
# Translate messages and save to output folder
translated_messages = translate_messages_to_folder(processed_messages, model, tokenizer, max_sequence_length, output_folder)
print("Messages translated and saved to the output folder.")

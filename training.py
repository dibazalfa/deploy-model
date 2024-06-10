# =============================================================================

import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords') # Add this line to download stopwords

lemmatizer = WordNetLemmatizer()

# reading the json.intense file
with open("intense.json") as file:
    intents = json.load(file)

# creating empty lists to store data
words = []
classes = []
documents = []
ignore_words = ["?", "!", ".", ","]

# Process each intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        if isinstance(pattern, str):
            word_list = word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, str(intent['tag'])))  # Convert intent tag to string
        else:
            print(f"Skipping non-string pattern: {pattern}")
    if str(intent['tag']) not in classes:  # Convert intent tag to string
        classes.append(str(intent['tag']))

# storing the root words or lemma
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words and w not in stopwords.words('english')]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

# Create an empty array for our output
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle the features and turn into np.array
random.shuffle(training)
print("Training data before conversion to numpy array:", training[:5])  # Print only the first 5 for brevity

# Convert training data to NumPy arrays
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# saving the words and classes list to binary files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# creating a Sequential machine learning model
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),),
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),
                activation='softmax'))

# compiling the model
#training the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(train_x, train_y, epochs=1000, batch_size=20, verbose=1, validation_split=0.2)


# saving the model
model.save("chatbotmodel.h5", hist)

# print statement to show the
# successful training of the Chatbot model
print("Yay!")

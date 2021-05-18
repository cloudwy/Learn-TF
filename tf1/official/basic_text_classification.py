"""
Dataset: IMDB - https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

Model:

Relative:
representation/word2vec.md
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Print version
print(tf.__version__)

# Load the data
#%%
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Explore the data
#%%
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

# Conver the integers back to words
#%%
word_index = imdb.get_word_index()
#前4个保留出来
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reserved_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reserved_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

# Preprocess - 用"<PAD>"在后端补齐
#%%
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

# Create the model
#%%
vocab_size = 10000

model = keras.Sequential()
# Embedding: input:(batch_size, sequence_length), output: (batch_size, sequence_length, output_dim)
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# Create a validation set
#%%
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model
#%%
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Plot train_loss and val_loss, train_acc and val_acc (review basic_regression.py)
#%%
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.plot(hist['epoch'], hist['acc'], label='Train Acc')
    plt.plot(hist['epoch'], hist['val_acc'], label='Val Acc')
    plt.legend()
    plt.show()
plot_history(history)

# Early Stop (review basic_regression.py)
#%%
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1,
                    callbacks=[early_stop])
plot_history(history)

# Evaluate
#%%
results = model.evaluate(test_data, test_labels, verbose=2)
print("evaluate result: ", results)

# Plot
#%%
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'go', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'go', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



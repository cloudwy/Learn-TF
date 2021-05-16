import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Print version
print(tf.__version__)

#%%
# Load Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore data
print("Shape of train images and labels: {}, {}".format(train_images.shape, train_labels.shape))
print("Shape of test images and labels: {}, {}".format(test_images.shape, test_labels.shape))

# Plot 1st image
plt.figure()
#plt.imshow(train_images[0])
#plt.imshow(train_images[0], cmap='gray')
plt.imshow(train_images[0], cmap='gray_r')
plt.colorbar()
plt.grid(False) #网格
#plt.show()
plt.close()

# Plot many images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)  #显示灰度图
    plt.xlabel(class_names[train_labels[i]])
plt.show()
plt.close()

# Preprocess
train_images = train_images / 255.0
test_images = test_images / 255.0


#%%
# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Create Checkpoint
log_path = "./ckpt"
#dt = datetime.now().strftime("%Y_%m_%d_%H_%M")
#log_path_dt = log_path+"/"+dt
os.makedirs(log_path, exist_ok=True)
ckpt_path = log_path + '/' + "model1.ckpt"
ckpt_dir = os.path.dirname(ckpt_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                 save_weights_only=False,
                                                 verbose=1)
"""
# Save checkpoints in the training

log_path = "./ckpt/model1"
os.makedirs(log_path, exist_ok=True)
ckpt_path = log_path + "/" + "cp-{epoch:02d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                 save_weights_only=False,
                                                 verbose=1,
                                                 period=2)
"""

# Train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_images, train_labels, epochs=6, callbacks=[cp_callback])

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("test loss: {}, test acc: {}".format(test_loss, test_acc))

# Make predictions
predictions = model.predict(test_images)
pred_class = np.argmax(predictions[0])
print("predicted label: {}, ground truth: {}".format(pred_class, test_labels[0]))


# Call a checkpoint
#%%
# Define a model
def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model

# load a checkpoint
model = create_model()
model_path = "./ckpt/model1.ckpt"
model.load_weights(model_path)
# load the latest checkpoint
#log_path = "./ckpt/model1"
#latest = tf.train.latest_checkpoint(log_path)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("test loss: {}, test acc: {}".format(loss, acc))
predictions = model.predict(test_images)
print("shape of predictions: ", predictions.shape)

# Image1: left(image, predicted_label, true_label), right(predictions_array)
def plot_image(i, predictions_array, true_label, img):
    """plot the ith image and label"""
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100 * np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    #predicted_label = np.argmax(predictions.array)
    #thisplot[predicted_label].set_color('red')
    #thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[0], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

#Image2: several images with their predictions
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows)) #figsize=(width，height)
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.show()



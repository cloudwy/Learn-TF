"""
Dataset: Auto MPG (https://archive.ics.uci.edu/ml/datasets/auto+mpg)
- 用来预测70年代末到80年代初汽车燃油效率的模型。
- 数据集提供了汽车相关描述，包含：气缸数，排量，马力以及重量等。

Model: Dense Neural Networks (3 layers)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Print version
print(tf.__version__)

# Load the data
#%%
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
init_shape = dataset.shape
print(dataset.tail()) #view the last 5 items
print(init_shape)

# Save the original dataset
#%%
raw_dataset.to_excel("auto-mpg.xls")

# Clean the data
#%%
print("num_rows with NaN values:\n", dataset.isna().sum()) #统计dataframe列中为NaN的行数
dataset = dataset.dropna() #删除含有NaN的行
print("shape before processing: {} and after processing:{} ".format(init_shape, dataset.shape))
origin = dataset.pop('Origin') #删除标签列"Origin"
print(dataset.tail())
# 转换成one-hot code
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())

# Split the data into train and test
#%%
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data
#%%
# view the joint distribution of a few pair of columns
#sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
#plt.show()
# view the overall statisticals
train_stats = train_dataset.describe()
print(train_stats)
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# Preprocess
#%%
# split features from labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
# normalization
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset) #需要将测试集映射到训练集的分布范围

# Create the Model
#%%
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()
model.summary()

# Try to predict on a batch
#%%
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

# Train the model
#%%
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print("epoch: {}, loss:{}".format(epoch, logs['loss']))

EPOCHS = 1000

history = model.fit(normed_train_data, train_labels,
                    epochs=EPOCHS, validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()])
print("properties of history: ", history.history.keys())
hist = pd.DataFrame(history.history)
print(hist.tail())
hist['epoch'] = history.epoch
print(hist.tail())

# Plot the history
#%%
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()

plot_history(history)

# Prevent overfitting - EarlyStopping
#%%
model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[early_stop, PrintDot()])
plot_history(history)

# Evaluate and Predict
#%%
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
test_predictions = model.predict(normed_test_data).flatten()
# plot scatter
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# plot histogram of error list
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()

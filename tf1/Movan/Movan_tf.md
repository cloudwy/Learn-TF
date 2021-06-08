# 莫烦PYTHON - Tensorflow

Website: 

https://mofanpy.com/tutorials/machine-learning/tensorflow/

Version：python3.6 + tf1.12



## 1. Structure

- 初始化变量不要忘记：sess.run(init)

```python
import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

#create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # one dimension, range[-1.0, 1.0]
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) # learning rate=0.5
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
#create tensorflow structure end

"""
sess = tf.Session()
sess.run(init) # Very important!

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
"""
with tf.Session() as sess:
    sess.run(init)
    # train
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(Weights), sess.run(biases))
```



## 2. Session

两种方法：

- sess = tf.Session()
- With tf.Session() as sess

```python
import tensorflow as tf

matrix1 = tf.constant([[3, 3]])  # 1x2
matrix2 = tf.constant([[2], [2]]) # 2x1

product = tf.matmul(matrix1, matrix2) # matrix multiply = np.dot(m1, m2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
# with can automatically close the Session
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
```



## 3. Variable

```python
import tensorflow as tf

state = tf.Variable(0, name='counter')
print(state.name)

# variable + constant = variable
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value) # replace state with new_value

# initialize variables - very important!
init = tf.initialize_all_variables()

# open session
with tf.Session() as sess:
    sess.run(init) # variables must be initialized!
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
```



## 4. Placeholder

- 运行到sess.run()时给feed_dict喂入数据

```python
import tensorflow as tf

input1 = tf.placeholder(tf.float32, [2, 2])
input2 = tf.placeholder(tf.float32, [2, 1])

output = tf.matmul(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [[1., 1.], [1., 1.]], input2: [[2.], [3.]]}))
```



## 5. Add Layer

```python
import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
```



## 6. Build Neural Network

```
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Define data
# (-1, 1)之间，分成300份，为1x300的矩阵，[:np,newaxis]转化为300x1的矩阵
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# Build a model
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1])) #按行求和

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #learning rate=0.1

init = tf.initialize_all_variables()

# Session
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i % 50 == 0:
            print("step:{}, loss:{}".format(i, sess.run(loss, feed_dict={xs:x_data, ys:y_data})))
```



## 7. Visualization

```python
# Session
with tf.Session() as sess:
    sess.run(init)

    fig = plt.figure()
    # 连续性的画图
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    # 不暂停继续plot
    plt.ion()
    #plt.show()
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 5 == 0:
            print("step:{}, loss:{}".format(i, sess.run(loss, feed_dict={xs: x_data, ys: y_data})))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
```



## 8. Tensorboard

- 视频中的tf版本小于1.12，因此会报错

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            #tf.histogram_summary(layer_name+'/weights', Weights)
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('bias'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            #tf.histogram_summary(layer_name + '/biases', biases)
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        #tf.histogram_summary(layer_name + '/outputs', outputs)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

# Define data
# (-1, 1)之间，分成300份，为1x300的矩阵，[:np,newaxis]转化为300x1的矩阵
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# inputs包含x_input和y_input
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# Build a model
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1])) #按行求和
    #tf.scatter_summary('loss', loss)
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #learning rate=0.1

init = tf.initialize_all_variables()

sess = tf.Session()
#merged = tf.merge_all_summaries() #将所有画图的summary打包
merged = tf.summary.merge_all()
#tf.train.SummaryWriter(<1.12版本)
writer = tf.summary.FileWriter("./logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data}) # Important!
        writer.add_summary(result, i)
```



## 9. Classification

- Dataset: MNIST

```
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pred = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print("step: {}, loss: {}, test_acc:{}".format(i,
                                          sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}),
                                          compute_accuracy(mnist.test.images, mnist.test.labels)))

```




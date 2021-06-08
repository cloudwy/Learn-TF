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

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
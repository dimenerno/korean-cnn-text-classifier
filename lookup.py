import tensorflow as tf

W = tf.Variable(tf.random_uniform([4, 6], -1.0, 1.0))
x = tf.constant(
    [[0, 1, 2], [0, 3, 2]])
embedded_chars = tf.nn.embedding_lookup(W, x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
sess = tf.Session()

sess.run(tf.global_variables_initializer())
print(sess.run(embedded_chars))
print(sess.run(tf.shape(embedded_chars)))
print(sess.run(embedded_chars_expanded))
print(sess.run(tf.shape(embedded_chars_expanded)))

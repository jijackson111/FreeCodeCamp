import tensorflow as tf

# Creating Tensors
string = tf.Variable("string", tf.string)
number = tf.Variable(3, tf.int64)
floating = tf.Variable(3.5, tf.float64)

# Rank/Degree of Tensors
rank1_tensor = tf.Variable(["Test", "yes", "hello"], tf.string)
rank2_tensor = tf.Variable([['test', 'ok'], ['test', 'yes']], tf.string)
print(tf.rank(rank1_tensor))
print(tf.rank(rank2_tensor))

# Shape of tensors
print('\n')
print(rank1_tensor.shape)
print(rank2_tensor.shape)

# Changing shape
tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [2, 3, 1])
tensor3 = tf.reshape(tensor2, [3, -1])
print('\n')
print(tensor1)
print('\n')
print(tensor2)
print('\n')
print(tensor3)

# Evaluating tensors
with tf.Session() as sess:
    tf.eval()
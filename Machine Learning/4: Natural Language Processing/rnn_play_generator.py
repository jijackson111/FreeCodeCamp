from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

# Import data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Reading the contents of the data
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print ('Length of text: {} characters'.format(len(text)))
print(text[:250])

# Creating a mapping from unique characters to incides:
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
def text_to_int(text):
  return np.array([char2idx[c] for c in text])
text_as_int = text_to_int(text)

# Print the encoded text
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

# Integer to Text function
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))

# Creating a stream of characters from text data
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Creating training examples
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Function to split sequences into input and ouput
def split_input_target(chunk): 
    input_text = chunk[:-1]
    target_text = chunk[1:] 
    return input_text, target_text
dataset = sequences.map(split_input_target) 

# Print
for x, y in dataset.take(2):
  print("\n\nEXAMPLE\n")
  print("INPUT")
  print(int_to_text(x))
  print("\nOUTPUT")
  print(int_to_text(y))
  
# Create training batches
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024
BUFFER_SIZE = 10000
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Building the model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

# Ask the model for a prediction on the first batch of training data
for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch) 
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
print(len(example_batch_predictions))
print(example_batch_predictions)

# Use the loss function to examine predictions
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
time_pred = pred[0]
print(len(time_pred))
print(time_pred)

# Sample the output distributionm and reshape the array tp determine the predicted character
sampled_indices = tf.random.categorical(pred, num_samples=1)
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)
predicted_chars

# Create a loss function
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Compiling the model
model.compile(optimizer='adam', loss=loss)

# Creating checkpoints for training
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Training
history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])

# Loading the model
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
checkpoint_num = 10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None]))

# Generating text
def generate_text(model, start_string):
  num_generate = 800
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = 1.0

  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])
      return (start_string + ''.join(text_generated))

# Displaying output
inp = input("Type a starting string: ")
print(generate_text(model, inp))
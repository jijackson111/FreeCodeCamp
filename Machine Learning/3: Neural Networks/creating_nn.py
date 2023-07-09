import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load dataset then split into testing and training (60000 images for training, 10000 for testing)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Observe the data
print(train_images.shape)
print(train_images[0,23,23])
print(train_labels[:10])

# Create class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Look at an image
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# Data preprocessing
train_images = train_images / 255.0  # Scaling our grayscale pixel values (0-255) to between 0 and 1
test_images = test_images / 255.0 # Smaller values will make it easier for the model to process

# Building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer: 784 neurons, flattern layer with a shape of (28,28)
    keras.layers.Dense(128, activation='relu'),  # hidden layer: 128 neurons, dense denotes that this layer will be fully connected
    keras.layers.Dense(10, activation='softmax') # output layer: also a dense layer, 10 neurons, each neuron represents the probability of a
    ])                                            # given image being one of the 10 different classes


# Compile the model (define the loss function, optimizer and metrics we would like to track)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10) 

# Evaluating the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
print('Test accuracy:', test_acc)

# Making predictions
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]

# Verifying predictions
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

# More verifying
def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)

# Show the image
def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()

# Get the number
def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
#import matplotlib.pyplot as plt
import tensorflow as tf

# Stop processing after the model hits a 99% accuracy rating.
class myCallback(tf.keras.callbacks.Callback):                      # Class created to handle the end of an Epoch
  def on_epoch_end(self, epoch, logs={}):                           # Takes in self, epoch, and logs
    if(logs.get('accuracy')>0.99):                                  # Accuracy is found within the logs input
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

# Initialize the callback function
callbacks = myCallback()

# Import the new dataset
mnist = tf.keras.datasets.mnist

# Load the data into the training and testing arrays
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualize the data
#plt.imshow(x_train[0])
#print(y_train[0])
#print(x_train[0])

# Normalize the gray scale images
x_train = x_train/255.0
x_test = x_test/255.0

# Define the model
model = tf.keras.models.Sequential([ tf.keras.layers.Flatten(input_shape = (28, 28)),           # Flatten to 1D array from 28x28 image
                                     tf.keras.layers.Dense(512, activation= tf.nn.relu),        # Find all possible results > 0, within 512 calculations
                                     tf.keras.layers.Dense(10, activation= tf.nn.softmax)])     # Narrow down to 0-9

# Compile with the loss and optimizer functions
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit to designed model in 5 epochs
model.fit(x_train, y_train, epochs = 10, callbacks = [callbacks])

# Determine which is the best choice.
model.evaluate(x_test, y_test)
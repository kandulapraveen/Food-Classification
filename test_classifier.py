import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time

start = time.time()

#Define Path
model_path = './Model/model.h5'
model_weights_path = './Model/weights.h5'
img_size = 512

#Define image 51
img_width, img_height = 512, 512
batch_size = 8

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)
# Data paths
test_data_dir = './Data/Val_data/'  # Replace with the path to the folder containing the test images

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create a data generator for test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' class_mode to get one-hot encoded labels
    shuffle=False              # Ensure that the order of predictions matches the order of test images
)

# Get the ground truth labels from the generator
ground_truth_labels = test_generator.classes
# print(ground_truth_labels)

# Load the trained model
# model = load_model('image_classification_model.h5')

# # Evaluate the model on test data
# loss, accuracy = model.predict(test_generator)


# Evaluate the model on test data
# model.evaluate(test_generator, ground_truth_labels)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  #print(result)
  answer = np.argmax(result)
  if answer == 0:
    print("Predicted: ApplePie")
    est_class = "ApplePie"

  elif answer == 1:
    print("Predicted: BagelSandwich")
    est_class = "BagelSandwich"
  elif answer == 2:
    print("Predicted: Bibimbop")
    est_class = "Bibimbop"
  elif answer == 3:
    print("Predicted: Bread")
    est_class = "Bread"
  elif answer == 4:
    print("Predicted: FriedRice")
    est_class = "FriedRice"

  elif answer == 5:
    print("Predicted: Pork")
    est_class = "Pork"

  print("Actual:", file)


  return answer, est_class

detected_labels = []
#Walk the directory for every image
accuracy = 0
temp = 0
for i, ret in enumerate(os.walk(test_data_dir)):
  for i, filename in enumerate(ret[2]):
    file_split = ret[0].split('/')
    
    # print(ret[0] + '/' + filename)
    result, est_class = predict(ret[0] + '/' + filename)
    if est_class in file_split:
      accuracy += 1
    temp += 1
#     detected_labels.append(result)
# print(detected_labels)
accuracy = accuracy/temp
print("accuracy:", accuracy)
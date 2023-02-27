import os
import tensorflow as tf
import numpy as np
from PIL import Image


# Load the trained model
model = tf.keras.models.load_model('flower_classifier_model.h5')

# Define a function to preprocess images before feeding them to the model
def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.asarray(image) / 255.
    image = np.expand_dims(image, axis=0)
    return image

# Define a function to classify an image and return the predicted class
def classify_image(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Walk through the flower_photos directory and classify all the images
image_classes = {}
for root, dirs, files in os.walk('static/flower_photos'):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            predicted_class = classify_image(image_path)
            image_classes[image_path] = predicted_class

# Save the image classes to a file
with open('image_classes.npy', 'wb') as f:
    np.save(f, image_classes)

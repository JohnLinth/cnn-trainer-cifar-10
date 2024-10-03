import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# change encoding to UTF-8 (keras didnt work with cp1252 encoding)
os.system('chcp 65001')
sys.stdout.reconfigure(encoding='utf-8')

# CIFAR-10 class names
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

model = load_model('cifar10_cnn_model7190.h5')

# function to preprocess the image for the model (resizes and normalizes jpg / png type images)
def preprocess_image(image_path):
    img = Image.open(image_path)

    # if the image has an alpha channel (PNG), convert to RGB to remove the alpha channel
    if img.mode == 'RGBA' or img.mode == 'LA':  # LA is grayscale with alpha
        img = img.convert('RGB')

    img = img.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255

    # reshape the image to add the batch dimension (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)

    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions)

    class_name = cifar10_classes[predicted_class]

    return class_name

image_path = 'donkey.jpg' 

# show image before prediction
img = Image.open(image_path)
plt.imshow(img)
plt.title("Uploaded Image")
plt.axis('off')
plt.show()

# predict
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")

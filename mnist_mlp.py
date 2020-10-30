'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

def load_keras_model():
    model = load_model('models') # from ./models directory
    return model

def predict_number(model, img, width, height):
    test_image = image.img_to_array(img)
    test_image = test_image.astype('float32')
    test_image = test_image.reshape(width, height)
    test_image /= 255
    test_image = test_image.reshape(1, width * height)
    result = model.predict(test_image, batch_size=1)
    return np.argmax(result)

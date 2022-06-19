import tensorflow as tf
from tensorflow import keras

from keras.models import load_model
from keras.preprocessing import image
import glob
import numpy as np
import pandas as pd

def find_car(input_dir, output_cars ="output.csv"):
  model = keras.models.load_model('./saved_model')
  input_dir = input_dir
  file_list = glob.glob(input_dir+"/*")
  files = []

  for file in file_list:
    file_n = file.replace("\\", "/").split('/')[-1]
    files.append(file_n)

  preds = []

  for file in file_list:
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    pred_b = tf.where(pred > 0.5, 1, 0)
    preds.append(pred_b.numpy().ravel())
  preds = np.array(preds).ravel()

  output = pd.DataFrame({'file': files, 'pred': preds})
  output['pred'] = output['pred'].astype(bool)
  output.to_csv(output_cars, index=False, header=False)

find_car('./output')
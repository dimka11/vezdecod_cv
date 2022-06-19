import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import load_model
from keras.preprocessing import image
from tensorflow import keras

def check_stop_signals(input_dir, output_file="output_color.csv"):
  model = keras.models.load_model('./saved_model_stop_signals')
  input_dir = input_dir
  file_list = glob.glob(input_dir+"/*")
  files = []
  preds = []

  for file in file_list:
    file_n = file.replace("\\", "/").split('/')[-1]
    files.append(file_n)

  for file in file_list:
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    pred_c = tf.argmax(pred, axis=1)
    preds.append(pred_c.numpy().ravel())
  
  preds = np.array(preds).ravel()
  output = pd.DataFrame({'file': files, 'pred': preds})
  output['pred'] = output['pred'].replace({0: 'no_cars', 1:'Off', 2:'On'})
  output.to_csv(output_file, index=False, header=False)
  return output

check_stop_signals('./output')
from flask import Flask,render_template,request

import os
import re
import gc
import numpy as np
import collections
from PIL import Image
from textwrap import wrap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, add


import os
import re
import gc
import numpy as np
import collections
from PIL import Image
from textwrap import wrap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, add

def word_for_id(integer, tokenizer):

	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	
	return None

def generate_desc(model, tokenizer, photo, max_len):
	
	start_text = 'startseq'

	for i in range(max_len):

		tokens = tokenizer.texts_to_sequences([start_text])[0]

		tokens = pad_sequences([tokens], maxlen=max_len)

		pred = model.predict([photo, tokens], verbose=0)

		pred = np.argmax(pred)

		word = word_for_id(pred, tokenizer)

		if word is None:
			break

		start_text += ' ' + word

		if word == 'endseq':
			break

	return start_text

xcep = Xception(include_top=False, pooling='avg')

with open('C:\\Users\\Rithwik Datta\\ML\\majorproject\\Flickr8k.token.txt', 'r') as f:
  all_desc = f.read().split('\n')

def clean_description(desc, stopwords):
    cleaned = desc.lower()
    cleaned = re.sub('[^a-z]',' ',cleaned)
    tokens = cleaned.split(' ')
    cleaned = ' '.join([w for w in tokens if w not in stopwords and len(w)>1])
    return cleaned

stopwords = ['is', 'an', 'a', 'the', 'was']
all_dict = dict()

for desc in all_desc:
  if len(desc) < 1:
    continue
  file_name, file_desc = desc.split('\t')[0].split('.')[0], desc.split('\t')[1]
  
  if file_name not in all_dict.keys():
    all_dict[file_name] = []

  cleaned_desc = clean_description(file_desc, stopwords)
  cleaned_desc = 'startseq ' + cleaned_desc + ' endseq'

  all_dict[file_name].append(cleaned_desc)

def get_vocabulary(dictionary):
  vocab = set()

  for desc_list in dictionary.values():
    for desc in desc_list:
      words = desc.split(' ')
      for word in words:
        vocab.add(word)

  return vocab

vocab = get_vocabulary(all_dict)


def create_list(dictionary):
  final_list = []

  for desc_list in dictionary.values():
    for desc in desc_list:
      final_list.append(desc)

  return final_list

def fit_tokenizer(dictionary):
  desc_list = create_list(dictionary)
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(desc_list)
  return tokenizer

def convert_to_input(tokens, pos, im_name, max_len, vocab_len, tokenizer, img_predictions):

  inp = tokens[:pos]
  out = tokens[pos]
  inp = pad_sequences(sequences=[inp], maxlen=max_len)[0]
  out = to_categorical(y=[out], num_classes=vocab_len, dtype='bool')[0]
  
  return img_predictions.get(im_name)[0], inp, out

def convert_all_to_input(dictionary, max_len, vocab_len, tokenizer, img_predictions):
  
  X_1 = list()
  X_2 = list()
  y = list()

  for im_name, descriptions in tqdm(dictionary.items()):
    if im_name in img_predictions.keys():
      for desc in descriptions:
          tokens = tokenizer.texts_to_sequences([desc])[0]
          for i in range(1, len(tokens)):
              _X_1, _X_2, _y = convert_to_input(tokens, i, im_name, max_len, vocab_len, tokenizer, img_predictions)
              X_1.append(_X_1)
              X_2.append(_X_2)
              y.append(_y)
  return np.array(X_1), np.array(X_2), np.array(y)

from tensorflow.keras.models import load_model
loaded_model = load_model("network.h5",compile=False)

predictions=np.load('my_file.npy',allow_pickle='TRUE').item()

tokenizer = fit_tokenizer(all_dict)
vocab_len = len(tokenizer.index_word) + 1
max_len = len(max(create_list(all_dict)))
cnn_len = predictions[list(predictions.keys())[0]].shape[1]


app= Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1

@app.route('/')
def index():
	return render_template("imagecaption.html", data="hey from flask")

@app.route("/prediction", methods=["POST"])
def prediction():

	global xcep

	image=request.files['img']
	image.save("static/img.jpeg")

	img_path = 'static/img.jpeg'
	img = Image.open(img_path)
	img2 = img.copy()
	img = img.resize((299,299))
	img = np.expand_dims(img, axis=0)
	img = img/127.5
	img = img - 1.0
	pred = xcep.predict(img)

	plt.figure(figsize=(10, 10))
	plt.imshow(img2)
	plt.axis('off')
	plt.show()

	caption = generate_desc(loaded_model, tokenizer, pred, 93)
	caption = caption.strip('startseq').strip('endseq')

	from gtts import gTTS 
	import os
	text=caption
	language='en'
	tld='co.in'

	speech = gTTS(text = text, lang = language, slow = False,tld='co.in')

	speech.save("static/text3.mp3")

	return render_template('prediction.html',caption=caption)

if __name__ =="__main__":
	app.run(debug=True)
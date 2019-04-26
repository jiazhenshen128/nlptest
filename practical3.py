# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:14:55 2019

@author: JiazhenShen
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:03:13 2019

@author: JiazhenShen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:16:19 2019

@author: JiazhenShen
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:06:38 2019

@author: JiazhenShen
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import zipfile
import lxml.etree
import re
from tensorflow import keras
import tensorflow as tf
from time import time
import numpy as np
import urllib

# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
# For now, we're only interested in the subtitle text, so let's extract that from the XML:
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = '\n\n\nNew'.join(doc.xpath('//content/text()'))
input_keywords = '\n'.join(doc.xpath('//keywords/text()'))
del doc

# Clean the inputs:
## Split the different subtitles, each one is a string. 
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
subtitles_strings_ted = input_text_noparens.split('\n\n\nNew')
del input_text_noparens

## label each subtitles
keywords_strings_ted = []
for line in input_keywords.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    keywords_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent) 
del m, line 
    
def keywords2labels(ks):
    ls = ['' for i in range(0,len(ks))]
    counter = 0
    for k in ks:
        if 'technology' in k:
            ls[counter] += 'T'
        else:
            ls[counter] += 'o'
        
        if 'entertainment' in k:
            ls[counter] += 'E'
        else:
            ls[counter] += 'o'
        
        if 'design' in k:
            ls[counter] += 'D'
        else:
            ls[counter] += 'o'
        
        counter += 1
    return ls

labels_ted = keywords2labels(keywords_strings_ted)
all_labels_ted = set(labels_ted)
ordered_all_labels_ted = sorted(all_labels_ted)
labels_dict = {l:i for i, l in enumerate(ordered_all_labels_ted)}
encoded_labels_ted = [labels_dict[i] for i in labels_ted]
encoded_labels_ted = keras.utils.to_categorical(encoded_labels_ted)
del labels_dict, ordered_all_labels_ted, all_labels_ted

# integer encode the documents and padding
## prepare tokenizer
t = keras.preprocessing.text.Tokenizer()
t.fit_on_texts(subtitles_strings_ted)
vocab_size = len(t.word_index) + 1
## integer encode the documents
encoded_subtitles_strings_ted = t.texts_to_sequences(subtitles_strings_ted)
## padding
max_length = max([len(a) for a in encoded_subtitles_strings_ted])
padded_subtitles_strings_ted =  keras.preprocessing.sequence.pad_sequences(encoded_subtitles_strings_ted , maxlen=max_length, padding='post')

# load the whole embedding into memory
embeddings_index = dict()
f = open('glove/glove.6B.50d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


x_train = padded_subtitles_strings_ted[:1585]
x_valid = padded_subtitles_strings_ted[1585:(1585+250)]
x_test = padded_subtitles_strings_ted[(1585+250):(1585+500)]

y_train = encoded_labels_ted[:1585]
y_valid = encoded_labels_ted[1585:(1585+250)]
y_test = encoded_labels_ted[(1585+250):(1585+500)]


# Model
## design the model
hidden_size = 3
batch_size = 50
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,50,weights=[embedding_matrix], input_length=max_length))
model.add(keras.layers.LSTM(hidden_size, batch_input_shape=(batch_size, 1, 50)))#, return_sequences=True))
#model.add(keras.layers.GlobalAveragePooling1D())
#model.add(keras.layers.Dense(3, activation='tanh'))
#model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(8, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
## summarize the model
print(model.summary())
## fit
tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))
history = model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=100,
                    validation_data=(x_valid, y_valid),
                    verbose=1,
                    callbacks=[tensorboard])

## test
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print('Accuracy: %f' % (accuracy*100))











        
    
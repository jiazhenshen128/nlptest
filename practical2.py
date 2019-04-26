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


### Split each one of subtitles as a list of sentences, each sentence is a string 
#counter = 0
#subtitles_sentences_strings_ted = subtitles_strings_ted[:]
#for input_sentences in subtitles_strings_ted:
#    sentences_strings_ted = []
#    for line in input_sentences.split('\n'):
#        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
#        sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent) 
#    subtitles_sentences_strings_ted[counter] = sentences_strings_ted
#    counter += 1
#del input_sentences, sentences_strings_ted, counter, line
#
#
### Split each one of sentences as a list of tokens
#subtitles_ted = subtitles_sentences_strings_ted[:]
#counter = 0
#for s in subtitles_sentences_strings_ted:
#    sentences_ted = []
#    for sent_str in s:
#        tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
#        sentences_ted.append(tokens)
#    subtitles_ted[counter] = sentences_ted
#    counter += 1
#del s, sent_str, counter, tokens

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
# integer encode the documents and padding
vocab_size = 3000
encoded_subtitles_strings_ted = [keras.preprocessing.text.one_hot(d, vocab_size) for d in subtitles_strings_ted]
max_length = max([len(a) for a in encoded_subtitles_strings_ted])
padded_subtitles_strings_ted =  keras.preprocessing.sequence.pad_sequences(encoded_subtitles_strings_ted , maxlen=max_length, padding='post')

x_train = padded_subtitles_strings_ted[:1585]
x_valid = padded_subtitles_strings_ted[1585:(1585+250)]
x_test = padded_subtitles_strings_ted[(1585+250):(1585+500)]

y_train = encoded_labels_ted[:1585]
y_valid = encoded_labels_ted[1585:(1585+250)]
y_test = encoded_labels_ted[(1585+250):(1585+500)]


# Model
## design the model
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_length=max_length))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(4, activation='relu'))
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











        
    
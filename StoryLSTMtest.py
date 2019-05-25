# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:26:45 2019

@author: hp
"""

from numpy import argmax
from pickle import load
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

def load_doc(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text

#list a predefined list of photo identifiers
def load_set(filename):
    doc =load_doc(filename)
    dataset=[]
    for line in doc.split('\n'):
        if(len(line)<1):
            continue
        #print(line.split('.'))
        iden=line.split('.')[0]
        #print(iden)
        dataset.append(iden)
    return set(dataset)

#load_set('D:\My_PROJECT\Flickr8k_text\Flickr_8k.trainImages.txt')    

def load_clean_descriptions(filename,dataset):
    doc=load_doc(filename)
    descriptions=dict()
    for line in doc.split('\n'):
        tokens=line.split()
        image_id,image_desc=tokens[0],tokens[1:]
        #skip image not in the set
        if image_id in dataset:
            #create list
            if image_id not in descriptions:
                descriptions[image_id]=[]
            desc = 'startseq '+' '.join(image_desc)+' endseq'
            descriptions[image_id].append(desc)
    return descriptions
import pickle as p
def load_photo_features(filename,dataset):
    all_features=p.load(open(filename,'rb'))
    features = {k:all_features[k] for k in dataset}
    return features

def to_lines(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines=to_lines(descriptions)
    tokenizer= Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
#length of the description with the most words
def max_length(descriptions):
    lines=to_lines(descriptions)
    return max(len(d.split()) for d in lines)
def word_for_id(integer,tokenizer):
    for word ,index in tokenizer.word_index.items():
        if index==integer:
            return word
        
#generate description of image        
def generate_desc(model,tokenizer,photo,max_length):
    in_text='startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence=pad_sequences([sequence],maxlen=max_length)
        yhat=model.predict([photo,sequence],verbose=0)
        #convert probability to integer
        yhat=argmax(yhat)
        #integer to word
        word=word_for_id(yhat,tokenizer)
        if word is None:
            break
        in_text+=' '+word
        if word=='endseq':
            break
    return in_text

def evaluate_model(model,descriptions,photos,tokenizer,max_length):
    actual,predicted=[],[]
    for key,desc_list in descriptions.items():
        yhat=generate_desc(model,tokenizer,photos[key],max_length)
        references=[d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    #calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# prepare tokenizer on train set

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions8k.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer,open('tokenizerLSTM.pkl','wb'))
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set

# load test set
filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions8k.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'D:\My_PROJECT\caption8kModel.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)    

#%%
"""test with new image"""
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

tokenizer=load(open('tokenizerLSTM.pkl','rb'))
max_length=34
filename = 'D:\My_PROJECT\caption8kModel.h5'
model = load_model(filename)

def extract_features(filename):
    model=load_model("D:\My_PROJECT\VGG")
    model.layers.pop()
    model=Model(inputs=model.inputs,outputs=model.layers[-1].output)
    print(model.summary())
    image=load_img(filename,target_size=(224,224))
    image=img_to_array(image)
        # reshape data for the model
        #1st param=no of sample
        #2nd param=rows
        #3rd param=columns
        #4th no of channels such as 3 for color image
    image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image=preprocess_input(image)
    features=model.predict(image,verbose=0)
    return features

photo=extract_features('example.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
label=' '.join(description.split()[1:-1])
print(label)

import cv2
image=cv2.imread('example.jpg')
cv2.putText(image,label, (10, 224), cv2.FONT_HERSHEY_SIMPLEX,
		         0.8, (0, 0, 255), 1,cv2.LINE_AA)

cv2.imshow('image', image)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:53:55 2019

@author: hp
"""

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model,load_model
from keras.utils.vis_utils import plot_model
from keras.applications.vgg16 import decode_predictions
# convert the probabilities to class labels

# extract features from each photo in the directory
def extract_features(directory):
    model=load_model("D:\My_PROJECT\VGG")
    model.layers.pop()
    model=Model(inputs=model.inputs,outputs=model.layers[-1].output)
    print(model.summary())
    features=dict()
    i=0
    for name in listdir(directory):
        filename=directory+'/'+name
        image=load_img(filename,target_size=(224,224))
        image=img_to_array(image)
        # reshape data for the model
        #1st param=no of sample
        #2nd param=rows
        #3rd param=columns
        #4th no of channels such as 3 for color image
        image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        image=preprocess_input(image)
        yhat=model.predict(image,verbose=0)
        
        image_id=name.split('.')[0]
        features[image_id]=yhat
        print(i,end=" ")
        i+=1
        
    return features

# extract features from all images
directory = 'D:\My_PROJECT\Flickr8k_Dataset\Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))

##read text file
descrip='D:\My_PROJECT\Flickr8k_text\Flickr8k.token.txt'
file=open(descrip,'r')
doc=file.read()
file.close()

def create_dict(doc):
    d=dict()
    for line in doc.split('\n'):
        tokens=line.split()
        if(len(line)<2):
            continue
        image_id,image_desc=tokens[0],tokens[1:]
        image_id=image_id.split('.')[0]
        image_desc=' '.join(image_desc)
        if image_id not in d:
            d[image_id]=[]
        d[image_id].append(image_desc)
        if len(d)==1000:
            break
    return d

descriptions=create_dict(doc)
print('Loaded: %d ' % len(descriptions)) 

import string
def clean_descriptions(desc):
    #translation table for removing punctuation
    table=str.maketrans('','',string.punctuation)
    print(table)
    for key,desc_list in desc.items():
        for i in range(len(desc_list)):
            d=desc_list[i]
            #tokenize
            d=d.split()
            #print(d)
            #convert to lower case
            d=[word.lower() for word in d]
            #remove punctuation
            d=[w.translate(table) for w in d]
            #remove hanging 's' and 'a'
            d=[w for w in d if len(w)>1]
            #remove tokens with numbers in them
            d=[w for w in d if w.isalpha()]
            #store as string
            desc_list[i]=' '.join(d)
clean_descriptions(descriptions) 

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))           

def save_descriptions(desc,filename):
    l=[]
    for key,desc_list in desc.items():
        for d in desc_list:
            l.append(key+' '+d)
    data='\n'.join(l) #convert into string by joining element with newline
    file=open(filename,'w')
    file.write(data)
    file.close()

save_descriptions(descriptions,'descriptions.txt')

#%%
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

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
        if len(dataset)==800:
            break
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
# train dataset

# load training dataset (6K)
filename = 'D:/My_PROJECT/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('D://My_PROJECT//features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
#%%
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]

def create_sequences(tokenizer,max_length,desc_list,photos):
    x1,x2,y=[],[],[]
    for desc in  desc_list:
        #encode the seq
        seq=tokenizer.texts_to_sequences([desc])[0]
        #split one sequence into multiple x,y pairs
        for i in range(1,len(seq)):
            in_seq,out_seq=seq[:i],seq[i]
            #pad input sequences
            in_seq=pad_sequences([in_seq],maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            x1.append(photos)
            x2.append(in_seq)
            y.append(out_seq)
    return array(x1),array(x2),array(y)
def define_model(vocab_size,max_length):
    # feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model

# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	# save model
	model.save('model_' + str(i) + '.h5')
model.save('D:\My_PROJECT\caption_train_model')    

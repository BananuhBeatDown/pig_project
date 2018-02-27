#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:29:20 2018

@author: matthew_green
"""

# from jpg_to_png_converter import png_converter
# from picture_array_converter import create_numpy_array_picture_dataset


from picture_array_creator import create_numpy_array_picture_dataset
from mylistdir import mylistdir
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import numpy as np
import csv

# Load real_test_dataset

directory = 'singular_pic_file'
picture = create_numpy_array_picture_dataset(directory)

# %%

shutil.copy2(directory + '/' + mylistdir(directory)[0], 'data_pictures_png_augmentation')

# %%
    
# Restore CNN model and make predicitons

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('metas/my_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('metas'))
    print("Model restored.")
    
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    logits = graph.get_tensor_by_name('logits:0')

    print("Initialized")
    test_prediction = sess.run(logits, feed_dict={x : picture, keep_prob: 1.})

# %%

# DISPLAY IMAGE AND LABEL:

def displaySequence():
    fig, ax = plt.subplots(1)
    plt.imshow(picture.reshape(64, 64, 3))
    plt.ion()
    plt.show()
    plt.pause(0.001)
    print ('Label : {}'.format(np.argmax(test_prediction)))
    input("Press [enter] to continue.")
    
displaySequence()
        
# %%

if np.argmax(test_prediction) == 0:
    fields = [mylistdir(directory)[0], 'paraffin', 'test', random.uniform(-99.66, -97.5), random.uniform(28.0, 29.0), mylistdir(directory)[0]]
elif np.argmax(test_prediction) == 1:
    fields = [mylistdir(directory)[0], 'no_mud', 'test', random.uniform(-99.66, -97.5), random.uniform(28.0, 29.0), mylistdir(directory)[0]]
else:
    fields = [mylistdir(directory)[0], 'corrosive', 'test', random.uniform(-99.66, -97.5), random.uniform(28.0, 29.0), mylistdir(directory)[0]]

# %%

with open('pic_data.csv', 'a') as f:
    writer = csv.writer(f, dialect='excel')
    writer.writerow([])
    writer.writerow(fields)
    




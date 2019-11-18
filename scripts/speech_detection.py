import keras
from keras import utils
import numpy as np

import tensorflow
import scipy
import os, shutil
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from six.moves import urllib
import sys
import tarfile
import os.path
from os import path
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import matplotlib.pyplot as plt
import subprocess
from timeit import default_timer as timer



from numpy  import array
from tensorflow.contrib.quantize.python import fold_batch_norms
from tensorflow.contrib.quantize.python import quantize
from tensorflow.python.framework import ops
print(tf.__version__)


tf.enable_eager_execution()

###   variables
MFCC=10
X_train_max=2.580736630323992
X_train_min=-6.401088510838375


########### FUNCTIONS

def predict(records, n):
	max = records[0]
	
	for i in range(1, n):
		if records[i] > max:
			max = records[i]
			
	index = record.index(max)
	
	return max, index

########################  Prepare
labels=['left', 'rigth', 'go', 'no', 'follow', 'up', 'stop', 'yes', 'down', 'forward']
	
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

tflite_model_file = "speech-model.tflite"
tflite_quant_model_file = "speech-model_quant_io.tflite"

X_test = X_test.reshape(X_test.shape[0], MFCC, 51, 1)

sounds = tf.cast(X_test, tf.float32)
urban_ds = tf.data.Dataset.from_tensor_slices(sounds).batch(1)


quant_sounds = tf.quantization.quantize(X_test, X_train_min, X_train_max, tf.quint8, mode="MIN_COMBINED", round_mode="HALF_AWAY_FROM_ZERO", name=None)
sounds_uint8 = tf.cast(quant_sounds[0], tf.uint8)
urban_ds_uint8 = tf.data.Dataset.from_tensor_slices(sounds_uint8).batch(1)

###################  Call interpreters

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_quant_model_file))
interpreter_quant.allocate_tensors()

####################
# Inference


runs = 1
print("Running inferencing for ", runs, "times.")

total_seen = 0
num_correct_float = 0
num_correct_quant = 0

for m in range(1,50):
	
	print("\nRunning number ", m)
	predict_number = m
	print(labels)
# Running Float inference
	i=0
	total_seen +=1
	
	for sound in urban_ds:
		
		if i == predict_number:
			break
		i = i+1
	
	interpreter.set_tensor(interpreter.get_input_details()[0]["index"], sound)

	if runs == 1:
		start = timer()	
		interpreter.invoke()
		end = timer()
		print("\n\nElapsed time running Non-quantized model is ", ((end - start)/runs)*1000, 'ms')
	else:
		start = timer()
		for i in range(0, runs):
			interpreter.invoke()
		end = timer()
		print("Elapsed time running Non-quantized model is ", ((end - start)/runs)*1000, 'ms')    
        

	predictions = interpreter.get_tensor( interpreter.get_output_details()[0]["index"])
	print("\nFloat predictions\n")
	print(predictions)
	
	class_prediction = predictions.tolist()
	
	for record in class_prediction:
		record
		
	records = array(record)
	
	n = len(records)
	
	class_predicted, indice = predict(records, n)
	
	real_words = y_test[predict_number]
	real_word = int(real_words)
	word_to_predict = labels[real_word]
	word_predicted = labels[indice]
	percentage = records[indice] * 100

	print("\nFLOATS")
	print("----------------")
	print("Predicted class is ", labels[indice], "with %5.2f" %(percentage) , "% accuracy")
	print("Real class is ", labels[real_word])
	print("-----------------\n\n")
	

# Running Integer Inference
	i=0
	
	for sound in urban_ds_uint8:
		
		if i == predict_number:
			break
		i = i+1
		
	interpreter_quant.set_tensor(interpreter_quant.get_input_details()[0]["index"], sound)

	if runs == 1:
		start = timer()
		interpreter_quant.invoke()
		end = timer()
		print("Elapsed time for one run of Quantized model is ", ((end - start)/runs)*1000, 'ms')
	else: 
		start = timer()
		for i in range(0, runs):
			interpreter.invoke()
		end = timer()
		print("Elapsed time running Quantized model is ", ((end - start)/runs)*1000, 'ms')


	quant_predictions = interpreter_quant.get_tensor(interpreter_quant.get_output_details()[0]["index"])
	print("\nQuantized predictions\n")
	print(quant_predictions)	
	
	
	class_prediction = quant_predictions.tolist()
	
	for record in class_prediction:
		record
		
	records = array(record)
	
	n = len(records)
	
	
	class_predicted, indice_quant = predict(records, n)
	print("quant records", records)
	word_quant_to_predict =labels[indice_quant]
	print(word_quant_to_predict)
	quant_percentage = (records[indice_quant] /255) * 100
	
	
		

	print("\nINTEGERS")
	print("----------------")
	print("Predicted class is ", labels[indice_quant], "with %5.2f" %(quant_percentage) , "% accuracy")	
	print("Real class is ", labels[real_word])
	print("-----------------")
	
	

	if word_predicted == word_to_predict:
		print("predicted correct word by float model")
		num_correct_float+=1

	if word_predicted == word_quant_to_predict:
		print("predicted correct word by quant model")
		num_correct_quant+=1		
	
print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
print("correct number of words predicted by Float model is ", num_correct_float)
print("Test Accuracy after %i images: %f" %(total_seen, float(num_correct_float) / float(total_seen)))
print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")




print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
print("correct number of words predicted by Quantized model is ", num_correct_quant)
print("Test Accuracy after %i images: %f" %(total_seen, float(num_correct_quant) / float(total_seen)))
print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")	



print("complete")



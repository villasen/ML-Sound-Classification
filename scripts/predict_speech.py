import numpy as np
import tensorflow
import os, shutil
import tensorflow as tf
import sys
import subprocess
from timeit import default_timer as timer


def predict(records, n):
	max = records[0]
	
	for i in range(1,n):
		if records[i] > max:
			max = records[i]
			
	index = record.index(max)
	
	return max, index
	

labels = ['down', 'right', 'stop', 'yes', 'forward', 'follow', 'left', 'up', 'no', 'go']




print('tensorflow version: ')
print(tf.__version__)


interpreter = tf.lite.Interpreter(model_path="/home/pi/machine_learning/speech/benchmark_10words/speech-model.tflite")
# Allocate memory. 
interpreter.allocate_tensors()

# get input and output tensors .
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

for m in range(1,10):
	for sound in urban_ds:
		if i == predict_number
			break
		i=i+1
		
		


interpreter.set_tensor(input_details[0]['index'], input_data)


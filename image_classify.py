import os
import pandas as pd
import json

pred_dict = {}
i = 0
directory = '/var/lib/neuraldata/neuraltalk2_flickr/benchmarkimages/'
for filename in os.listdir(directory):
	i += 1
	print ("Iter:", i)
	script = 'python /var/lib/neuraldata/tensorflowmods/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/imagenet/classify_image.py --image_file ' + directory
	command = script + filename
	top_5 = os.popen(command).read()
	#print (type(top_5))
	#print (top_5)
	top_5_labels = top_5.split ("\n")
	print ("The top five labels are as follows"+ "\n") 
	print (top_5_labels)
	print ("Assigning filename and classification to dictionary...")
	pred_dict[filename] = top_5_labels
	print ("Assigned!")
print ("The mapped dictionary is as follows:")
print (pred_dict)
print ("Dictionary Created with mapping Filename -> Top 5 Label List")
json.dump(pred_dict, open('pred_dict.json', 'w')) 

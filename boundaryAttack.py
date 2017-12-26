import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import numba

from optparse import OptionParser
import wget
import tarfile
import os
import sys
import cv2
import time
import shutil
import math

# Add the path to the tensorflow models repository
sys.path.append("/mnt/BoundaryAttack/models/research/slim")
sys.path.append("/mnt/BoundaryAttack/models/research/slim/nets")

import inception_resnet_v2
import resnet_v1
import nasnet.nasnet as nasnet

import sys
if sys.version_info[0] == 3:
	print ("Using Python 3")
	import pickle as cPickle
else:
	print ("Using Python 2")
	import cPickle

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-m", "--model", action="store", type="string", dest="model", default="NAS", help="Model to be used for Cross-Layer Pooling")

parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=1001, help="Number of classes in the dataset")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in the image")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=1, help="Batch size")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs", help="Directory for saving logs")
parser.add_option("--valFileName", action="store", type="string", dest="valFileName", default="/mnt/BoundaryAttack/data/filenames.txt", help="List of image files present in the dataset")
parser.add_option("--classNamesFile", action="store", type="string", dest="classNamesFile", default="/mnt/BoundaryAttack/data/class_names.txt", help="File containing the names of all classes")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

numAdverserialUpdates = 100
sigma = 0.5
epsilon = 0.5
assert (options.batchSize == 1)

baseDir = os.getcwd()
with open(options.classNamesFile, 'r') as imageClassNamesFile:
    classDict = eval(imageClassNamesFile.read())

# Offset the dictionary
classDictExtended = {0: 'background'}
for cls in classDict:
	classDictExtended[cls+1] = classDict[cls]

# Load the model
if options.model == "ResNet":
	print ("Loading ResNet-152")
	resnet_checkpoint_file = checkpointFileName = os.path.join(baseDir, 'resnet_v1_152.ckpt')
	if not os.path.isfile(resnet_checkpoint_file):
		# Download file from the link
		url = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'
		filename = wget.download(url)

		# Extract the tar file
		tar = tarfile.open(filename)
		tar.extractall()
		tar.close()

	options.imageHeight = options.imageWidth = 224

elif options.model == "IncResV2":
	print ("Loading Inception ResNet v2")
	inc_res_v2_checkpoint_file = checkpointFileName = os.path.join(baseDir, 'inception_resnet_v2_2016_08_30.ckpt')
	if not os.path.isfile(inc_res_v2_checkpoint_file):
		# Download file from the link
		url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
		filename = wget.download(url)

		# Extract the tar file
		tar = tarfile.open(filename)
		tar.extractall()
		tar.close()

	options.imageHeight = options.imageWidth = 299

elif options.model == "NAS": 
	print ("Loading NASNet")
	nas_checkpoint_file = checkpointFileName = os.path.join(baseDir, 'model.ckpt')
	if not os.path.isfile(nas_checkpoint_file + '.index'):
		# Download file from the link
		url = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
		filename = wget.download(url)

		# Extract the tar file
		tar = tarfile.open(filename)
		tar.extractall()
		tar.close()

	# Update image sizes
	options.imageHeight = options.imageWidth = 331

else:
	print ("Error: Unknown model selected")
	exit(-1)

# Define params
IMAGENET_MEAN = [123.68, 116.779, 103.939] # RGB

# Reads an image from a file, decodes it into a dense tensor
def _parse_function(filename, label):
	image_string = tf.read_file(filename)
	img = tf.image.decode_jpeg(image_string)

	img = tf.image.resize_images(img, [options.imageHeight, options.imageWidth])

	img.set_shape([options.imageHeight, options.imageWidth, options.imageChannels])
	img = tf.cast(img, tf.float32) # Convert to float tensor
	return img, filename, label

def loadDataset(currentDataFile):
	print ("Loading data from file: %s" % (currentDataFile))
	dataClasses = {}
	with open(currentDataFile) as f:
		imageFileNames = f.readlines()
		imNames = []
		imLabels = []
		for imName in imageFileNames:
			imName = imName.strip().split(' ')
			imNames.append(imName[0])
			currentLabel = int(imName[1])
			imLabels.append(currentLabel)

			if currentLabel not in dataClasses:
				dataClasses[currentLabel] = 1
			else:
				dataClasses[currentLabel] += 1

		imNames = tf.constant(imNames)
		imLabels = tf.constant(imLabels)

	numClasses = len(dataClasses)
	numFiles = len(imageFileNames)
	print ("Dataset loaded")
	print ("Files: %d | Classes: %d" % (numFiles, numClasses))
	# print (dataClasses)

	dataset = tf.contrib.data.Dataset.from_tensor_slices((imNames, imLabels))
	dataset = dataset.map(_parse_function)
	# dataset = dataset.shuffle(buffer_size=numFiles)
	dataset = dataset.batch(options.batchSize)

	return dataset, numClasses

# A vector of filenames
valDataset, _ = loadDataset(options.valFileName)
valIterator = valDataset.make_initializable_iterator()

global_step = tf.train.get_or_create_global_step()

with tf.name_scope('Model'):
	# Data placeholders
	inputBatchImages, inputBatchImageNames, inputBatchLabels = valIterator.get_next()
	print ("Data shape: %s" % str(inputBatchImages.get_shape()))

	# Data placeholders
	inputBatchImagesPlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, options.imageHeight, options.imageWidth, options.imageChannels], name="inputBatchImages")

	if options.model == "IncResV2":
		scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImagesPlaceholder)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

		# Create model
		arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
		with slim.arg_scope(arg_scope):
			logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages, is_training=False)

	elif options.model == "ResNet":
		if options.useImageMean:
			imageMean = tf.reduce_mean(inputBatchImagesPlaceholder, axis=[1, 2], keep_dims=True)
			print ("Image mean shape: %s" % str(imageMean.shape))
			processedInputBatchImages = inputBatchImagesPlaceholder - imageMean
		else:
			channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImagesPlaceholder)
			for i in range(options.imageChannels):
				channels[i] -= IMAGENET_MEAN[i]
			processedInputBatchImages = tf.concat(axis=3, values=channels)
			print (processedInputBatchImages.shape)

		# Create model
		arg_scope = resnet_v1.resnet_arg_scope()
		with slim.arg_scope(arg_scope):
			# logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=options.trainModel, num_classes=numClasses)
			logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=False)

	elif options.model == "NAS":
		scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImagesPlaceholder)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

		# Create model
		arg_scope = nasnet.nasnet_large_arg_scope()
		with slim.arg_scope(arg_scope):
			# logits, end_points = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=options.trainModel, num_classes=numClasses)
			logits, end_points = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=False, num_classes=options.numClasses)

	else:
		print ("Error: Unknown model selected")
		exit(-1)


def computeDistance(firstImage, secondImage):
	dist = np.mean(np.square(firstImage - secondImage))
	return dist

# @numba.jit
def sampleAdverserialExample(sess, originalImage, adverserialImage):
	# Sample adverserial update from a iid distribution with range [0, 1)
	adverserialUpdate = np.random.rand(batchImages.shape[1], batchImages.shape[2], batchImages.shape[3])
	
	# Clip the values of the update vector so that the first constraint holds
	adverserialUpdate[(adverserialImage + adverserialUpdate) > 255.0] = 255.0 - adverserialImage[(adverserialImage + adverserialUpdate) > 255.0]
	adverserialUpdate[(adverserialImage + adverserialUpdate) < 0.0] = -adverserialImage[(adverserialImage + adverserialUpdate) < 0.0] # Since adverserial update cannot be negative, therefore, the adverserial image must be negative

	# Scale the magnitude of the adverial update so that the second constraint holds
	dist = computeDistance(originalImage, adverserialImage)
	normOfAdverserialUpdate = np.linalg.norm(adverserialUpdate)
	alpha = (sigma * dist) / normOfAdverserialUpdate
	adverserialUpdate = adverserialUpdate * alpha

	# Second condition: Norm of adverserial update = sigma * distance between original image and new adverserial example
	secondCondMet = (normOfAdverserialUpdate == sigma * dist)
	if not secondCondMet:
		print ("Error: Second condition failed")
		print ("Norm:", normOfAdverserialUpdate, "| Sigma:", sigma, "| Distance:", dist)
		exit (-1)

	# Third condition: Difference between d(original image, adverserial example) and d(original image, updated adverserial example) should be equal to epsilon * d(original image, adverserial example)
	distOriginalAdverserial = computeDistance(originalImage, adverserialImage)
	distOriginalUpdatedAdverserial = computeDistance(inputImage, (adverserialImage + adverserialUpdate))
	thirdCondMet = (distOriginalAdverserial - distOriginalUpdatedAdverserial == epsilon * distOriginalAdverserial)

	newAdverserialImage = adverserialImage + adverserialUpdate
	newAdverserialImagePredictedLabel = sess.run(predictedClass, feed_dict={inputBatchImagesPlaceholder: np.expand_dims(newAdverserialImage, axis=0)})
	imageStillAdverserial = adverserialImagePredictedLabels[0] != predictedLabels[0]

	return newAdverserialImage


# Create the predicted class node
predictedClass = tf.argmax(end_points['Predictions'], axis=1)

# Initializing the variables
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# Perform boundary attack
with tf.Session(config=config) as sess:
	# Initialize all vars
	sess.run(init)
	sess.run(init_local)

	# Load the imagenet pre-trained model
	print ("Restoring model from file: %s" % checkpointFileName)
	restorer = tf.train.Saver()
	restorer.restore(sess, checkpointFileName)
	print ("Model restored!")

	sess.run(valIterator.initializer)
	
	while True:
		start_time = time.time()
		# Obtain the image
		[batchImages, batchImageNames, batchLabels] = sess.run([inputBatchImages, inputBatchImageNames, inputBatchLabels])
		predictedLabels = sess.run(predictedClass, feed_dict={inputBatchImagesPlaceholder: batchImages})
		predictedLabels -= 1 # Compensate for the offset in the classes (1001)	
		
		print ("Predictions shape:", predictedLabels.shape)
		print ("Batch image names:", batchImageNames)

		# Start creation of adverserial example
		inputImage = batchImages[0, :, :, :]
		for i in range(batchLabels.shape[0]):
			print ("Correct: %s | Original label: %s | Predicted Label: %s" % (str(batchLabels[i] == predictedLabels[i]), classDict[batchLabels[i]], classDict[predictedLabels[i]]))
			
			# Initialize the adverserial example
			while True:
				adverserialImage = np.random.rand(batchImages.shape[1], batchImages.shape[2], batchImages.shape[3]) * 256.0
				print ("Adverserial image shape:", adverserialImage.shape)

				adverserialImagePredictedLabels = sess.run(predictedClass, feed_dict={inputBatchImagesPlaceholder: np.expand_dims(adverserialImage, axis=0)})
				adverserialImagePredictedLabels -= 1 # Compensate for the offset in the classes (1001)
				if adverserialImagePredictedLabels[0] != predictedLabels[0]:
					print ("Image successfully initialized!")
					print ("Image # %d | Original image prediction: %s | Adverserial image prediction: %s" % (i, classDict[predictedLabels[i]], classDict[adverserialImagePredictedLabels[i]]))
					break

			# Perform updates on the adverserial example
			for i in range(numAdverserialUpdates):
				adverserialImage = sampleAdverserialExample(sess, inputImage, adverserialImage)

			cv2.imshow("Input image", inputImage[:, :, ::-1].astype(np.uint8))
			cv2.imshow("Adverserial image", adverserialImage[0].astype(np.uint8))

			char = cv2.waitKey()
			if (char == ord('q')):
				print ("Process terminated by user!")
				exit(-1)

		duration = time.time() - start_time

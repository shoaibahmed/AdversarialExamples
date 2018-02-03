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

# Clone the repository if not already existent
if not os.path.exists("./models/research/slim"):
	print ("Cloning TensorFlow models repository")
	from git import Repo # gitpython
	Repo.clone_from("https://github.com/tensorflow/models.git", ".")
	print ("Repository sucessfully cloned!")

# Add the path to the tensorflow models repository
sys.path.append("./models/research/slim")
sys.path.append("./models/research/slim/nets")

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
parser.add_option("--writeImagesToLogDir", action="store_true", dest="writeImagesToLogDir", default=False, help="Whether to write images to directory")
parser.add_option("--showImages", action="store_true", dest="showImages", default=False, help="Whether to show adverserial images once they are computed")

parser.add_option("--minDomainValue", action="store", type="float", dest="minDomainValue", default=0.0, help="Minimum value that can occur within the input domain")
parser.add_option("--maxDomainValue", action="store", type="float", dest="maxDomainValue", default=255.0, help="Maximum value that can occur within the input domain")

# Attack parameters
parser.add_option("--numAdverserialUpdates", action="store", type="int", dest="numAdverserialUpdates", default=100, help="Number of attack iterations to be performed")
parser.add_option("--numDirectionsToExplore", action="store", type="int", dest="numDirectionsToExplore", default=25, help="Number of random directions to be explored per iteration of the attack")
parser.add_option("--sigma", action="store", type="float", dest="sigma", default=0.5, help="Sigma to be used for the attack (defines the relative size of the perturbation)")
parser.add_option("--epsilon", action="store", type="float", dest="epsilon", default=0.5, help="Epsilon to be used for the attack (defines the relative amount by which the distance between the original and the perturbed image is reduced)")
parser.add_option("--stepAdaptationRate", action="store", type="float", dest="stepAdaptationRate", default=1.5, help="The adaptation rate by which the epsilon and sigma is adjusted")
parser.add_option("--convergenceThreshold", action="store", type="float", dest="convergenceThreshold", default=1e-7, help="Threshold for convergence of attack")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

assert (options.batchSize == 1)
if not (options.writeImagesToLogDir or options.showImages):
	print ("Error: No output selected. Either enable the --showImages flag to show images directly on the screen or --writeImagesToLogDir to write images to log directory.")
	exit(-1)

if options.writeImagesToLogDir:
	if os.path.exists(options.logsDir):
		print ("Removing previous directory")
		shutil.rmtree(options.logsDir)

	print ("Creating logs directory")
	os.mkdir(options.logsDir)

currentEpsilon = 0.0
currentSigma = 0.0

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

# Create the predicted class node
predictedClass = tf.argmax(end_points['Predictions'], axis=1)

# Helper methods
def computeDistance(firstImage, secondImage, normalized=True):
	dist = np.mean(np.square(firstImage - secondImage))
	if normalized:
		dist = dist / (options.maxDomainValue * options.maxDomainValue)
	return dist

def computeOriginalImageDirection(firstImage, secondImage):
	originalImageVector = firstImage - secondImage
	originalImageNorm = np.linalg.norm(originalImageVector)
	originalImageDirection = originalImageVector / originalImageNorm
	return originalImageVector, originalImageDirection, originalImageNorm

def generateCandidates(originalImage, adverserialUpdate, originalImageVector, originalImageDirection, originalImageNorm):
	# Project the point onto the sphere
	projection = np.vdot(adverserialUpdate, originalImageDirection)
	adverserialUpdate -= projection * originalImageDirection
	adverserialUpdate *= currentSigma * originalImageNorm / np.linalg.norm(adverserialUpdate)

	D = 1.0 / np.sqrt(currentSigma**2 + 1)
	direction = adverserialUpdate - originalImageVector
	sphericalCandidate = originalImage + D * direction
	sphericalCandidate = np.clip(sphericalCandidate, options.minDomainValue, options.maxDomainValue)

	# Add perturbation in the direction of the source
	newOriginalImageDirection = originalImage - sphericalCandidate
	newOriginalImageNorm = np.linalg.norm(newOriginalImageDirection)

	# Length of the vector assuming spherical candidate to be exactly on the sphere
	lengthOfSphericalCandidate = currentEpsilon * originalImageNorm
	deviation = newOriginalImageNorm - originalImageNorm
	lengthOfSphericalCandidate += deviation
	lengthOfSphericalCandidate = np.maximum(0, lengthOfSphericalCandidate) # Keep only positive numbers
	lengthOfSphericalCandidate = lengthOfSphericalCandidate / newOriginalImageNorm

	candidate = sphericalCandidate + lengthOfSphericalCandidate * newOriginalImageDirection
	candidate = np.clip(candidate, options.minDomainValue, options.maxDomainValue)

	return candidate, sphericalCandidate

def updateStepSizes(sphericalSuccessProbability, stepSuccessProbability):
	global currentEpsilon
	global currentSigma

	sphericalBasedParameterChange = True
	if sphericalSuccessProbability > 0.5:
		print ('Boundary too linear, increasing steps!')
		currentSigma *= options.stepAdaptationRate
		currentEpsilon *= options.stepAdaptationRate
	elif sphericalSuccessProbability < 0.2:
		print ('Boundary too non-linear, decreasing steps!')
		currentSigma /= options.stepAdaptationRate
		currentEpsilon /= options.stepAdaptationRate
	else:
		sphericalBasedParameterChange = False

	stepBasedParameterChange = True
	if stepSuccessProbability > 0.5:
		print ('Success rate too high, increasing original image steps!')
		currentEpsilon *= options.stepAdaptationRate
	elif stepSuccessProbability < 0.2:
		print ('Success rate too low, decreasing original image steps!')
		currentEpsilon /= options.stepAdaptationRate
	else:
		stepBasedParameterChange = False

	if sphericalBasedParameterChange or stepBasedParameterChange:
		print ("Step parameters updated | Spherical step size (sigma): %f | Original image step size (epsilon): %f" % (currentSigma, currentEpsilon))
	else:
		print ('Retaining previous step parameters')

# @numba.jit
def sampleAdverserialExample(sess):
	# Reset step sizes
	global currentSigma
	global currentEpsilon
	currentEpsilon = options.epsilon
	currentSigma = options.sigma

	# Obtain the image
	[batchImages, batchImageNames, batchLabels] = sess.run([inputBatchImages, inputBatchImageNames, inputBatchLabels])
	predictedLabels = sess.run(predictedClass, feed_dict={inputBatchImagesPlaceholder: batchImages})
	predictedLabels -= 1 # Compensate for the offset in the classes (1001)	

	# Remove the batch dimension
	originalImage = batchImages[0, :, :, :]
	batchImageNames = batchImageNames[0].decode("utf-8")
	batchLabels = batchLabels[0]
	predictedLabels = predictedLabels[0]
	fileName = batchImageNames[batchImageNames.rfind(os.sep)+1:batchImageNames.rfind('.')] # Extract the root name
	print ("Image name: %s | Correct: %s | Original label: %s | Predicted Label: %s" % (batchImageNames, str(batchLabels == predictedLabels), classDict[batchLabels], classDict[predictedLabels]))
	
	# Initialize the adverserial example
	while True:
		adverserialImage = np.random.rand(originalImage.shape[0], originalImage.shape[1], originalImage.shape[2]) * 256.0

		adverserialImagePredictedLabels = sess.run(predictedClass, feed_dict={inputBatchImagesPlaceholder: np.expand_dims(adverserialImage, axis=0)})
		adverserialImagePredictedLabels -= 1 # Compensate for the offset in the classes (1001)
		adverserialImagePredictedLabels = adverserialImagePredictedLabels[0] # Remove batch dim
		if adverserialImagePredictedLabels != predictedLabels:
			print ("Image successfully initialized!")
			print ("Original image label: %s | Original image prediction: %s | Adverserial image prediction: %s" % 
				(classDict[batchLabels], classDict[predictedLabels], classDict[adverserialImagePredictedLabels]))
			break

	# Perform updates on the adverserial example
	initialAdverserialImage = adverserialImage.copy().astype(np.uint8)[:, :, ::-1]

	# Iterate over the number of iterations to be performed
	convergenceStep = options.numAdverserialUpdates - 1
	for step in range(options.numAdverserialUpdates):
		# Variables for statistics
		numSuccessSpherical = 0
		numSuccessSteps = 0
		numTotalAttempts = options.numDirectionsToExplore

		# Compute unit vector pointing from the original image to the adverserial image
		originalImageVector, originalImageDirection, originalImageNorm = computeOriginalImageDirection(originalImage, adverserialImage)

		numStepToConvergence = 0
		distance = computeDistance(originalImage, adverserialImage)
		print ("Distance between original image and adverserial image: %f" % distance)

		# Check if adverserial attack converged
		# if distance < options.convergenceThreshold:
		if currentEpsilon < options.convergenceThreshold:
			print ("Attack converged after %d iterations" % step)
			convergenceStep = step - 1
			adverserialImagePredictedLabel = sess.run(predictedClass, feed_dict={inputBatchImagesPlaceholder: np.expand_dims(adverserialImage, axis=0)})
			newCandidatePredictedLabel = adverserialImagePredictedLabel[0]
			break

		newAdverserialImageDistance = float('Inf')
		newAdverserialImage = None

		for directionIteration in range(options.numDirectionsToExplore):
			# Sample adverserial update from a iid distribution with range [0, 1)
			adverserialUpdate = np.random.rand(originalImage.shape[0], originalImage.shape[1], originalImage.shape[2])
			
			# Generate the candidates based on the input
			candidate, sphericalCandidate = generateCandidates(originalImage, adverserialUpdate, originalImageVector, originalImageDirection, originalImageNorm)

			# Check if the spherical candidate is adverserial
			sphericalCandidatePredictedLabel = sess.run(predictedClass, feed_dict={inputBatchImagesPlaceholder: np.expand_dims(sphericalCandidate, axis=0)})
			sphericalCandidatePredictedLabel = sphericalCandidatePredictedLabel[0] # Remove batch dim
			isSphericalCandidateAdverserial = sphericalCandidatePredictedLabel != batchLabels
			isCandidateAdverserial = False
			if isSphericalCandidateAdverserial:
				numSuccessSpherical += 1
				candidatePredictedLabel = sess.run(predictedClass, feed_dict={inputBatchImagesPlaceholder: np.expand_dims(candidate, axis=0)})
				candidatePredictedLabel = candidatePredictedLabel[0] # Remove batch dim
				isCandidateAdverserial = candidatePredictedLabel != batchLabels
			else:
				# Perform next iteration
				continue

			if isCandidateAdverserial:
				# newAdverserialImageDistance = computeDistance(originalImage, newAdverserialImage)
				currentDist = computeDistance(originalImage, candidate)
				if currentDist < newAdverserialImageDistance:
					numSuccessSteps += 1
					newAdverserialImageDistance = currentDist
					newAdverserialImage = candidate
					newCandidatePredictedLabel = candidatePredictedLabel

		# Handle the found adverserial example
		if newAdverserialImage is not None:
			if newAdverserialImageDistance >= distance:
				print ("Warning: Current adverserial image's distance is greater than the previous distance")
			else:
				absoluteImprovement = distance - newAdverserialImageDistance
				relativeImprovement = absoluteImprovement / distance
				print ("Absolute improvement: %f | Relative improvement: %f" % (absoluteImprovement, relativeImprovement))

				# Update the variables
				adverserialImage = newAdverserialImage
				distance = newAdverserialImageDistance

		# Update the alpha and epsilon based on the success probability
		sphericalSuccessProbability = float(numSuccessSpherical) / numTotalAttempts
		stepSuccessProbability = float(numSuccessSteps) / numTotalAttempts
		print ("Step: %d | Total attempts: %d | Successful attempts (spherical): %d | Successful attempts (candidate): %d | Spherical success probability: %f | Step success probability: %f" % 
			(step, numTotalAttempts, numSuccessSpherical, numSuccessSteps, sphericalSuccessProbability, stepSuccessProbability))
		updateStepSizes(sphericalSuccessProbability, stepSuccessProbability)

		adverserialImageOut = adverserialImage[:, :, ::-1].astype(np.uint8)
		cv2.putText(adverserialImageOut, 'Initial prediction (adverserial): %s' % (classDict[adverserialImagePredictedLabels]), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
		cv2.putText(adverserialImageOut, 'Final prediction (adverserial): %s' % (classDict[newCandidatePredictedLabel]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
		cv2.putText(adverserialImageOut, 'Distance from original image: %s' % (distance), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

		if options.writeImagesToLogDir:
			cv2.imwrite(os.path.join(options.logsDir, fileName + '-adverserial-' + str(step) + '.png'), adverserialImageOut)

	print ("Original image label: %s | Original image prediction: %s | Initial adverserial image prediction: %s | Final adverserial image prediction: %s" % 
		(classDict[batchLabels], classDict[predictedLabels], classDict[adverserialImagePredictedLabels], classDict[newCandidatePredictedLabel]))

	inputImageOut = originalImage[:, :, ::-1].astype(np.uint8)
	cv2.putText(inputImageOut, 'Original class: %s' % (classDict[batchLabels]), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
	cv2.putText(inputImageOut, 'Predicted class: %s' % (classDict[predictedLabels]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

	if options.showImages:
		cv2.imshow("Input image", inputImageOut)
		cv2.imshow("Adverserial image", initialAdverserialImage)	
		cv2.imshow("Updated adverserial image", adverserialImageOut)

	# Write the images to the log
	if options.writeImagesToLogDir:
		cv2.imwrite(os.path.join(options.logsDir, fileName + '-input.png'), inputImageOut)
		cv2.imwrite(os.path.join(options.logsDir, fileName + '-initial.png'), initialAdverserialImage)
		cv2.imwrite(os.path.join(options.logsDir, fileName + '-adverserial-' + str(convergenceStep) + '.png'), adverserialImageOut)

	char = cv2.waitKey()
	if (char == ord('q')):
		print ("Process terminated by user!")
		exit(-1)

	return adverserialImage, newCandidatePredictedLabel


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
		# Start creation of adverserial example
		start_time = time.time()
		adverserialImage, adverserialImagePredictedLabels = sampleAdverserialExample(sess)
		duration = time.time() - start_time

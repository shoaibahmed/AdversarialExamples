import os
import sys
import csv
import shutil
import time

import cv2
import sklearn
import numpy as np
from tqdm import tqdm

import gzip
import tensorflow as tf
import foolbox

from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod, LBFGS, CarliniWagnerL2, SPSA, MadryEtAl, ElasticNetMethod, DeepFool, FastFeatureAdversaries, MomentumIterativeMethod, BasicIterativeMethod, SaliencyMapMethod

try:
	import cPickle as pickle # Python2
except ModuleNotFoundError:
	import pickle # Python3

cleverHansAttacks = ["FGSM", "LBFGS", "CarliniWagnerL2", "SPSA", "MadryEtAl", "ElasticNet", "DeepFool", "MomentumIterative", "BasicIterative", "SaliencyMap"]
foolBoxAttacks = ["BoundaryAttack", "LBFGS", "SaliencyMap"]

# dataset
dataset = "fashion_mnist"  # "mnist", "toyseq", "normal", "german_signs", "fashion_mnist"
attackMethods = ["CarliniWagnerL2"]
if len(sys.argv) > 1:
	attackMethods = sys.argv[1].split(';')

for attackMethod in attackMethods:	
	assert ((attackMethod in cleverHansAttacks) or (attackMethod in foolBoxAttacks)), "Error: Unknown attack (%s)" % (attackMethod)
print ("Attack methods: %s" % attackMethods)

#################### Utility Classes (Start) ####################

class Data:
	def __init__(self):
		self._idx_train = []
		self._X_train = []
		self._X_train_adv = []
		self._y_train = []
		self._y_train_adv = []

		self._idx_test = []
		self._X_test = []
		self._X_test_adv = []
		self._y_test = []
		self._y_test_adv = []

	def print_data_shape(self):
		print ("Train | X: %s | y: %s" % (str(self._X_train.shape), str(self._y_train.shape)))
		print ("Test | X: %s | y: %s" % (str(self._X_test.shape), str(self._y_test.shape)))


class CleverHansModel(Model):
	def __init__(self, numClasses, trainModel, scope=None, **kwargs):
		Model.__init__(self, scope=scope, nb_classes=numClasses, hparams=locals())
		self.trainModel = trainModel
		self.scope = "" if scope is None else scope

	def fprop(self, x, **kwargs):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			logits, endPoints = createNetwork(x, trainModel=self.trainModel, returnEndpoints=True)

		return {self.O_LOGITS: logits,
				self.O_PROBS: tf.nn.softmax(logits=logits)}
		# return endPoints # For version 2.1.0


class Config:
	def __init__(self):
		self.seed = 0
		self.out_frac = 0
		self.ad_experiment = False

		# "mnist" parameters (-1 stands for all the rest)
		self.mnist_normal = 0
		self.mnist_outlier = -1

		# "cifar10" parameters (-1 stands for all the rest)
		self.cifar10_normalize_mode = "fixed value"  # "per channel", "per pixel"
		self.cifar10_normal = 1
		self.cifar10_outlier = -1

		self.numClasses = 43
		self.numTrainingEpochs = 15
		self.numBatchSize = 250
		self.weightDecay = 1e-4
		self.useCleverHans = True
		self.numAdversarialExamples = -1
		self.attackIterations = 100
		self.epsilon = 1e-2
		self.advExamplesClass = 14 # Stop sign class
		self.useProgressBar = True

		########## Train CNN ##########
		if dataset == "GTSRB":
			self.cache_path = "../cache-gtsrb/"
			self.checkpoint_path = "../checkpoints-gtsrb/"

			self.adv_path = "../data/GTSRB/adv/"
			self.data_path = "../data/GTSRB/"
		elif dataset == "fashion_mnist":
			self.numClasses = 10
			self.advExamplesClass = 5 # Sandal (https://github.com/zalandoresearch/fashion-mnist)

			self.cache_path = "../cache-fashion-mnist/"
			self.checkpoint_path = "../checkpoints-fashion-mnist/"

			self.adv_path = "../data/fashion-mnist/adv/"
			self.data_path = "../data/fashion-mnist/"
		else:
			print("Error: Unknown dataset (%s)" % dataset)
			exit(-1)

#################### Utility Class (End) ####################


#################### Utility Functions (Start) ####################

def addConv(net, kernelSize, numFilters, layerID, addPooling=False, useBN=True, training=False):
	net = tf.layers.conv2d(net, filters=numFilters, kernel_size=(kernelSize, kernelSize), strides=(1, 1), padding='SAME', \
				kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Cfg.weightDecay), activation=None, name='conv' + layerID)
	
	if useBN:
		net = tf.layers.batch_normalization(net, training=training, name='bn' + layerID)
	
	# net = tf.nn.relu(net, name='relu' + layerID)
	net = tf.nn.leaky_relu(net, name='relu' + layerID)

	if addPooling:
		net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='pool' + layerID)

	return net

# Define the network
def createNetwork(X, trainModel, batchNorm=True, returnEndpoints=False):
	endPoints = {}
	# Conv1 (32 x 32)
	net = addConv(X, 3, 128, '1', addPooling=True, useBN=batchNorm, training=trainModel)
	if returnEndpoints:
		endPoints['conv1'] = net

	# Conv2 (16 x 16)
	net = addConv(net, 3, 256, '2', addPooling=True, useBN=batchNorm, training=trainModel)
	if returnEndpoints:
		endPoints['conv2'] = net

	# Conv3 (8 x 8)
	net = addConv(net, 3, 256, '3', addPooling=True, useBN=batchNorm, training=trainModel)
	if returnEndpoints:
		endPoints['conv3'] = net

	# Conv4 (4 x 4)
	net = addConv(net, 3, 512, '4', addPooling=True, useBN=batchNorm, training=trainModel)
	if returnEndpoints:
		endPoints['conv4'] = net

	# FC layers
	net = tf.layers.flatten(net, name="flatten")
	net = tf.layers.dense(net, units=512, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Cfg.weightDecay), name="fc1")

	if trainModel:
		net = tf.nn.dropout(net, 0.5) # Drop 50% of the neurons

	if batchNorm:
		net = tf.layers.batch_normalization(net, training=trainModel, name='bn_fc1')

	net = tf.nn.leaky_relu(net, name='relu_fc1')
	if returnEndpoints:
		endPoints['fc1'] = net

	net = tf.layers.dense(net, units=256, activation=None, name="fc2")

	if trainModel:
		net = tf.nn.dropout(net, 0.5) # Drop 50% of the neurons
	
	if batchNorm:
		net = tf.layers.batch_normalization(net, training=trainModel, name='bn_fc2')

	net = tf.nn.leaky_relu(net, name='relu_fc2')
	if returnEndpoints:
		endPoints['fc2'] = net

	net = tf.layers.dense(net, units=Cfg.numClasses, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Cfg.weightDecay), name="logits")
	if returnEndpoints:
		endPoints['logits'] = net

	if returnEndpoints:
		return net, endPoints
	else:
		return net

def testNetworkPerformance(sess, Cfg, data, predictions, xPlaceholder):
	systemPredictions = []
	# Compute test accuracy
	for (X, y, datasetName) in [(data._X_train, data._y_train, "Train"), (data._X_test, data._y_test, "Test")]:
		datasetPredictions = None
		numIterations = int(np.ceil(X.shape[0] / float(Cfg.numBatchSize)))
		for iteration in range(numIterations):
			startingIdx = iteration * Cfg.numBatchSize
			endingIdx = min((iteration + 1) * Cfg.numBatchSize, X.shape[0])
			currentPredictions = sess.run([predictions], feed_dict={xPlaceholder: X[startingIdx:endingIdx]})[0]
			if datasetPredictions is None:
				datasetPredictions = currentPredictions
			else:
				datasetPredictions = np.append(datasetPredictions, currentPredictions, axis=0)

		print ("Predictions shape: %s" % str(datasetPredictions.shape))
		datasetAccuracy = np.mean(datasetPredictions == y)
		print ("Dataset: %s | Accuracy: %f" % (datasetName, datasetAccuracy))
		systemPredictions.append(datasetPredictions)

	return systemPredictions

def generatePrediction(sess, x, predictions, xPlaceholder):
	currentPrediction = sess.run([predictions], feed_dict={xPlaceholder: np.expand_dims(x, axis=0)})[0]
	return int(currentPrediction[0])

def loadDataFromCSVFile(prefix, gtFile, normalize=True):
	images = [] # images
	labels = [] # corresponding labels

	gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
	gtReader.next() # skip header
	# loop over all images in current annotations file
	for row in gtReader:
		x1 = int(row[3])
		y1 = int(row[4])
		x2 = int(row[5])
		y2 = int(row[6])
		img = cv2.imread(prefix + row[0], cv2.IMREAD_COLOR)  # the 1st column is the filename
		img = img[x1:x2, y1:y2]  # remove border of 10% around sign
		img = cv2.resize(img, (32, 32))  # resize to 32x32

		if normalize:
			# Contrast normalization
			img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
			img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
			img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

			# Normalization
			img = img / 255.0

		img = img[:, :, ::-1] # flip from BGR to RGB
		assert (np.min(img) >= 0.0) and (np.max(img) <= 1.0)

		images.append(img.astype(np.float32))
		labels.append(int(row[7]))  # the 8th column is the label
	gtFile.close()

	return np.array(images), np.array(labels)

def readTrafficSigns(rootpath, which_set="train", label=-1):
	'''
	Reads traffic sign data for German Traffic Sign Recognition Benchmark.
	'''

	if which_set == "train":
		if label == -1:
			X = None
			y = None
			for c in range(Cfg.numClasses):
				dir_path = rootpath + "Final_Training/Images"
				prefix = dir_path + '/' + format(c, '05d') + '/'  # subdirectory for class
				gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
				X_c, y_c = loadDataFromCSVFile(prefix, gtFile)
				if X is None:
					X = X_c
					y = y_c
				else:
					X = np.concatenate((X, X_c), axis=0)
					y = np.concatenate((y, y_c), axis=0)

		else:
			dir_path = rootpath + "Final_Training/Images"
			prefix = dir_path + '/' + format(label, '05d') + '/'  # subdirectory for class
			gtFile = open(prefix + 'GT-' + format(label, '05d') + '.csv')  # annotations file
			X, y = loadDataFromCSVFile(prefix, gtFile)

	if which_set == "test":
		dir_path = rootpath + "Final_Test/Images"
		prefix = dir_path + '/'
		gtFile = open(prefix + '/' + 'GT-final_test.csv')  # annotations file
		X, y = loadDataFromCSVFile(prefix, gtFile)

	print ("Set: %s | X shape: %s | y shape: %s" % (which_set, str(X.shape), str(y.shape)))
	return X, y

def load_mnist(path, kind='train'):
	"""Load MNIST data from `path`"""
	labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
	images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

	with gzip.open(labels_path, 'rb') as lbpath:
		labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

	with gzip.open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

	images = images.reshape([-1, 28, 28, 1])
	print("Before normalization | Min value:", np.min(images), "Max value:", np.max(images))

	# Normalize values
	images = images.astype(np.float32) / 255.0

	print("After normalization | Min value:", np.min(images), "Max value:", np.max(images))
	return images, labels

def loadData(dataset):
	# Load dataset
	loadRawData = True
	startTime = time.time()

	if dataset == "gtsrb":
		pickleFileName = os.path.join(Cfg.cache_path, dataset + ".pickle")

		if os.path.exists(Cfg.cache_path):
			if os.path.exists(pickleFileName):
				# Load data
				loadRawData = False
				print ("Loading data from cache")
				with open(pickleFileName, "rb") as pickleFile:
					data = pickle.load(pickleFile)
		else:
			os.makedirs(Cfg.cache_path)

		if loadRawData:
			print ("Loading raw data")
			X_train, y_train = readTrafficSigns(Cfg.data_path, which_set="train")
			X_test, y_test = readTrafficSigns(Cfg.data_path, which_set="test")
			data = Data()
			data._X_train = X_train
			data._y_train = y_train
			data._X_test = X_test
			data._y_test = y_test

			# Dump data to cache
			with open(pickleFileName, "wb") as pickleFile:
				pickle.dump(data, pickleFile, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		data = Data()
		X_train, y_train = load_mnist(Cfg.data_path, kind='train')
		X_test, y_test = load_mnist(Cfg.data_path, kind='t10k')

		data._X_train = X_train
		data._y_train = y_train
		data._X_test = X_test
		data._y_test = y_test

	endTime = time.time()
	print ("Data loaded successfully! Time elapsed: %s seconds" % str(endTime - startTime))
	print ("Train | X shape: %s | y shape: %s" % (str(data._X_train.shape), str(data._y_train.shape)))
	print ("Test | X shape: %s | y shape: %s" % (str(data._X_test.shape), str(data._y_test.shape)))
	
	return data

#################### Utility Functions (End) ####################


Cfg = Config()

trainModel = False
if not os.path.exists(Cfg.checkpoint_path):
	os.makedirs(Cfg.checkpoint_path)
	trainModel = True
else:
	trainModel = not os.path.exists(os.path.join(Cfg.checkpoint_path, dataset + ".meta"))

if trainModel:
	print ("Checkpoint not found. Training model from scratch!")
else:
	print ("Checkpoint found. Evaluating pretrained model!")

data = loadData(dataset)
numTrainingIterations = int(np.ceil(data._X_train.shape[0] / float(Cfg.numBatchSize)))

if trainModel:
	print ("Number of training examples: %d | Number of examples per batch: %d | Number of iterations per epoch: %d" % (data._X_train.shape[0], Cfg.numBatchSize, numTrainingIterations))

	if dataset == "gtsrb":
		xPlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="xPlaceholder")
	else:  # Fashion MNIST
		xPlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="xPlaceholder")

	#xPlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, 32 if dataset == "gtsrb" else 28, 32 if dataset == "gtsrb" else 28, 3], name="xPlaceholder")
	yPlaceholder = tf.placeholder(dtype=tf.int64, shape=[None], name="yPlaceholder")

	# Create the network
	logits = createNetwork(xPlaceholder, trainModel=True)
	predictions = tf.argmax(logits, axis=1, name="predictions")

	correctPredictions = tf.equal(predictions, yPlaceholder)
	accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32), name="accuracy")

	# Define the loss
	xeLoss = tf.losses.sparse_softmax_cross_entropy(labels=yPlaceholder, logits=logits, scope="xeLoss")
	regLoss = tf.reduce_sum(tf.losses.get_regularization_losses(), name="regLoss")
	loss = tf.add(xeLoss, regLoss, name="loss")

	# Create the optimizer
	globalStep = tf.Variable(0, trainable=False)
	learningRate = tf.train.exponential_decay(1e-2, globalStep, 500, 0.5, staircase=True) # 0.01 is the starting learning rate
	
	optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
	updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(updateOps):
		trainOp = optimizer.minimize(loss, global_step=globalStep)
	saver = tf.train.Saver(var_list=tf.global_variables())

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print ("Initiating network training")
		# Train the network
		for epoch in range(Cfg.numTrainingEpochs):
			print ("Starting epoch # %d" % epoch)

			# Shuffle data
			data._X_train, data._y_train = sklearn.utils.shuffle(data._X_train, data._y_train)

			for iteration in range(numTrainingIterations):
				startingIdx = iteration * Cfg.numBatchSize
				endingIdx = min((iteration + 1) * Cfg.numBatchSize, data._X_train.shape[0])
				X = data._X_train[startingIdx:endingIdx]
				y = data._y_train[startingIdx:endingIdx]
				currentXELoss, currentRegLoss, currentLoss, currentAccuracy, _ = sess.run([xeLoss, regLoss, loss, accuracy, trainOp], feed_dict={xPlaceholder: X, yPlaceholder: y})
				if iteration % 50 == 0:
					print ("Iteration: %d | Cross-Entropy: %f | Regularizaiton Loss: %f | Total Loss: %f | Accuracy: %f" % (iteration, currentXELoss, currentRegLoss, currentLoss, currentAccuracy))

		# Save the model
		saver.save(sess, os.path.join(Cfg.checkpoint_path, dataset))

		testNetworkPerformance(sess, Cfg, data, predictions, xPlaceholder)

numTrainingExamples = np.sum(data._y_train == Cfg.advExamplesClass) if Cfg.advExamplesClass >= 0 else data._y_train.shape[0]
numTestExamples = np.sum(data._y_test == Cfg.advExamplesClass) if Cfg.advExamplesClass >= 0 else data._y_test.shape[0]
print ("Attack class ID: %d | Training examples found: %d | Test examples found: %d" % (Cfg.advExamplesClass, numTrainingExamples, numTestExamples))

for attackMethod in attackMethods:
	if attackMethod in foolBoxAttacks:
		Cfg.useCleverHans = False
	else:
		Cfg.useCleverHans = True

	advDirName = dataset + "_" + str(Cfg.numAdversarialExamples) + "_Examples_" + (("CleverHans_" + attackMethod) if Cfg.useCleverHans else ("Foolbox_" + attackMethod)) + "_" + str(Cfg.attackIterations) + "_Iterations_" + str(Cfg.epsilon) + "_Eps"
	advFileName = os.path.join(Cfg.adv_path, advDirName, advDirName + ".pickle")
	advDirName = os.path.join(Cfg.adv_path, advDirName)
	advImagesFileName = os.path.join(advDirName, "Images.pickle")
	advLabelsFileName = os.path.join(advDirName, "Labels.pickle")
	advSetFileName = os.path.join(advDirName, "Set.pickle")
	print ("Adversarial pickle file name: %s" % advFileName)

	if not os.path.exists(os.path.join(advFileName)):
		# Reset previous graph
		tf.reset_default_graph()
	
		print ("Performing attack: %s" % (attackMethod))

		if dataset == "gtsrb":
			xPlaceholder = tf.placeholder(dtype=tf.float32, shape=[1 if attackMethod == "SPSA" else None, 32, 32, 3], name="xPlaceholder")  # Hardcoded 1 for SPSA
		else:  # Fashion MNIST
			xPlaceholder = tf.placeholder(dtype=tf.float32, shape=[1 if attackMethod == "SPSA" else None, 28, 28, 1], name="xPlaceholder")  # Hardcoded 1 for SPSA

		yPlaceholder = tf.placeholder(dtype=tf.int64, shape=[1 if attackMethod == "SPSA" else None], name="yPlaceholder")

		if attackMethod == "SPSA":
			Cfg.numBatchSize = 1

		if Cfg.useCleverHans:
			model = CleverHansModel(numClasses=Cfg.numClasses, trainModel=False)
			logits = model.get_logits(xPlaceholder)
		else:
			logits = createNetwork(xPlaceholder, trainModel=False)

		predictions = tf.argmax(logits, axis=1)
		saver = tf.train.Saver(var_list=tf.global_variables())

		# Start the attack
		dataAdv = Data()

		if Cfg.advExamplesClass >= 0:
			invSetDict = {"Train": 0, "Test": 1}
			advImages = []
			advLabels = []
			advSet = []

		with tf.Session() as sess:
			print ("Initiating network evaluation")
			sess.run(tf.global_variables_initializer())

			saver.restore(sess, os.path.join(Cfg.checkpoint_path, dataset))
			systemPredictions = testNetworkPerformance(sess, Cfg, data, predictions, xPlaceholder)

			if attackMethod == "FGSM":
				attackParams = { 'eps': Cfg.epsilon * 10.0, 'clip_min': 0., 'clip_max': 1. }
			if attackMethod == "LBFGS":
				Cfg.useCleverHans = False # Use Foolbox for LBFGS
				attackParams = { 'max_iterations': 1, 'clip_min': 0., 'clip_max': 1. }
			if attackMethod == "CarliniWagnerL2":
				attackParams = { 'max_iterations': Cfg.attackIterations, 'clip_min': 0., 'clip_max': 1. }
			if attackMethod == "MadryEtAl":
				attackParams = { 'eps_iter': Cfg.attackIterations, 'eps': Cfg.epsilon * 10.0, 'clip_min': 0., 'clip_max': 1. }
			if attackMethod == "ElasticNet":
				attackParams = { 'max_iterations': Cfg.attackIterations, 'clip_min': 0., 'clip_max': 1. }
			if attackMethod == "DeepFool":
				attackParams = { 'max_iter': Cfg.attackIterations, 'clip_min': 0., 'clip_max': 1. }
			if attackMethod == "MomentumIterative":
				attackParams = { 'eps_iter': Cfg.attackIterations, 'eps': Cfg.epsilon * 3.0, 'clip_min': 0., 'clip_max': 1. }
			if attackMethod == "BasicIterative":
				attackParams = { 'eps_iter': Cfg.attackIterations, 'eps': Cfg.epsilon * 10.0, 'clip_min': 0., 'clip_max': 1. }
			if attackMethod == "SaliencyMap":
				Cfg.useCleverHans = False # Use Foolbox for Saliency map (Clever Hans is too slow)
				attackParams = { 'theta': Cfg.epsilon, 'clip_min': 0., 'clip_max': 1. }

			if Cfg.useCleverHans:
				# Initialize CleverHans
				print ("Initializing CleverHans")
				if attackMethod == "FGSM":
					print ("Using FGSM attack method!")
					attack = FastGradientMethod(model=model, sess=sess)
				if attackMethod == "LBFGS":
					print ("Using LBFGS attack method!")
					attack = LBFGS(model=model, sess=sess)
				if attackMethod == "CarliniWagnerL2":
					print ("Using Carlini and Wagner attack method!")
					attack = CarliniWagnerL2(model=model, sess=sess)
				if attackMethod == "SPSA":
					print ("Using SPSA attack method!")
					attack = SPSA(model=model, sess=sess)
				if attackMethod == "MadryEtAl":
					print ("Using Madry et al. attack method!")
					attack = MadryEtAl(model=model, sess=sess)
				if attackMethod == "ElasticNet":
					print ("Using Elastic Net attack method!")
					attack = ElasticNetMethod(model=model, sess=sess)
				if attackMethod == "DeepFool":
					print ("Using Deep Fool attack method!")
					attack = DeepFool(model=model, sess=sess)
				if attackMethod == "MomentumIterative":
					print ("Using Momentum Iterative attack method!")
					attack = MomentumIterativeMethod(model=model, sess=sess)
				if attackMethod == "BasicIterative":
					print ("Using Basic Iterative attack method!")
					attack = BasicIterativeMethod(model=model, sess=sess)
				if attackMethod == "SaliencyMap":
					print ("Using Saliency Map attack method!")
					attack = SaliencyMapMethod(model=model, sess=sess)

				if attackMethod == "SPSA":
					adversarialOp = attack.generate(x=xPlaceholder, y=yPlaceholder, epsilon=Cfg.epsilon * 5.0, num_steps=Cfg.attackIterations)
				elif attackMethod == "LBFGS":
					adversarialOp = attack.generate(x=xPlaceholder, y_target=tf.one_hot(yPlaceholder, depth=Cfg.numClasses), **attackParams)
				else:
					adversarialOp = attack.generate(x=xPlaceholder, y=tf.one_hot(yPlaceholder, depth=Cfg.numClasses), **attackParams)
			else:
				# Initialize Foolbox
				print ("Initializing FoolBox")
				model = foolbox.models.TensorFlowModel(images=xPlaceholder, logits=logits, bounds=(0, 1))
				if attackMethod == "BoundaryAttack":
					print ("Using Boundary attack method!")
					attack = foolbox.attacks.BoundaryAttack(model)
				if attackMethod == "SaliencyMap":
					print ("Using Saliency Map attack method!")
					attack = foolbox.attacks.SaliencyMapAttack(model)
				if attackMethod == "LBFGS":
					print ("Using LBFGS attack method!")
					attack = foolbox.attacks.LBFGSAttack(model)

			for (X, y, yPred, datasetName) in [(data._X_train, data._y_train, systemPredictions[0], "Train"), (data._X_test, data._y_test, systemPredictions[1], "Test")]:
				if Cfg.useProgressBar:
					progressBar = tqdm(total=numTrainingExamples if datasetName == "Train" else numTestExamples, position=0)

				numAdvExamplesGenerated = 0
				for iterator in range(X.shape[0]):
					if Cfg.useProgressBar:
						progressBar.update(1)
					
					# Only consider examples which were correctly classified by the network
					if y[iterator] != yPred[iterator]:
						continue

					# Consider only the examples from a particular class if Cfg.advExampleClass >= 0
					if (Cfg.advExamplesClass >= 0) and (y[iterator] != Cfg.advExamplesClass):
						continue

					if Cfg.useCleverHans:
						adversarial = sess.run([adversarialOp], feed_dict={xPlaceholder: np.expand_dims(X[iterator], axis=0), yPlaceholder: np.expand_dims(y[iterator], axis=0)})[0]
						adversarial = adversarial[0] # Remove batch dim
					else:
						if attackMethod == "BoundaryAttack":
							# 1000 iterations instead of 100
							adversarial = attack(input_or_adv=X[iterator], label=int(y[iterator]), iterations=Cfg.attackIterations * 50, internal_dtype=np.float32, log_every_n_steps=100, verbose=False)
						elif attackMethod == "SaliencyMap":
							adversarial = attack(input_or_adv=X[iterator], label=int(y[iterator]), max_iter=Cfg.attackIterations, theta=Cfg.epsilon * 50.0)
						elif attackMethod == "LBFGS":
							adversarial = attack(input_or_adv=X[iterator], label=int(y[iterator]), maxiter=Cfg.attackIterations, epsilon=Cfg.epsilon)
						else:
							adversarial = attack(input_or_adv=X[iterator], label=int(y[iterator]))

						if adversarial is None:
							continue

					prediction = generatePrediction(sess, adversarial, predictions, xPlaceholder)

					# If adversarial example creation was successful
					if y[iterator] != prediction:
						if datasetName == "Train":
							dataAdv._idx_train.append(iterator)
							dataAdv._X_train.append(X[iterator])
							dataAdv._X_train_adv.append(adversarial)
							dataAdv._y_train.append(y[iterator])
							dataAdv._y_train_adv.append(prediction)

						elif datasetName == "Test":
							dataAdv._idx_test.append(iterator)
							dataAdv._X_test.append(X[iterator])
							dataAdv._X_test_adv.append(adversarial)
							dataAdv._y_test.append(y[iterator])
							dataAdv._y_test_adv.append(prediction)

						else:
							print ("Error: Dataset name not found (%s)!" % datasetName)
							exit (-1)

						# Add data to the updated format
						if Cfg.advExamplesClass >= 0:
							advImages.append(X[iterator])
							advImages.append(adversarial)
							advLabels.append(0)
							advLabels.append(1)
							advSet.append(invSetDict[datasetName])
							advSet.append(invSetDict[datasetName])

						numAdvExamplesGenerated += 1
						if Cfg.useProgressBar:
							tqdm.write("%d adversarial examples generated" % numAdvExamplesGenerated)
						else:
							print ("Adversarial example generated successfully for the corresonding %d image in the %s dataset (%d out of %d)" % (iterator, datasetName, \
									numAdvExamplesGenerated, Cfg.numAdversarialExamples if Cfg.numAdversarialExamples > 0 else X.shape[0]))

						if (Cfg.numAdversarialExamples > 0) and (numAdvExamplesGenerated >= Cfg.numAdversarialExamples):
							print ("Warning: Stopping adversarial examples generated. Maximum Capacity reached. %d examples out of %d permissable examples generated." % (numAdvExamplesGenerated, Cfg.numAdversarialExamples))
							break

				# Convert to numpy arrays
				if datasetName == "Train":
					dataAdv._idx_train = np.array(dataAdv._idx_train)
					dataAdv._X_train = np.array(dataAdv._X_train)
					dataAdv._X_train_adv = np.array(dataAdv._X_train_adv)
					dataAdv._y_train = np.array(dataAdv._y_train)
					dataAdv._y_train_adv = np.array(dataAdv._y_train_adv)

				elif datasetName == "Test":
					dataAdv._idx_test = np.array(dataAdv._idx_test)
					dataAdv._X_test = np.array(dataAdv._X_test)
					dataAdv._X_test_adv = np.array(dataAdv._X_test_adv)
					dataAdv._y_test = np.array(dataAdv._y_test)
					dataAdv._y_test_adv = np.array(dataAdv._y_test_adv)

				else:
					print ("Error: Dataset name not found (%s)!" % datasetName)
					exit (-1)

				if Cfg.useProgressBar:
					progressBar.close()

			# Save the generated adversarial examples
			if not os.path.exists(advDirName):
				os.makedirs(advDirName)
			
			with open(advFileName, "wb") as pickleFile:
				pickle.dump(dataAdv, pickleFile, protocol=pickle.HIGHEST_PROTOCOL)

			if Cfg.advExamplesClass >= 0:
				advImages = np.array(advImages)
				advLabels = np.array(advLabels)
				advSet = np.array(advSet)

				# Dump the data in the compatible format
				with open(advImagesFileName, "wb") as pickleFile:
					pickle.dump(advImages, pickleFile, protocol=pickle.HIGHEST_PROTOCOL)
				with open(advLabelsFileName, "wb") as pickleFile:
					pickle.dump(advLabels, pickleFile, protocol=pickle.HIGHEST_PROTOCOL)
				with open(advSetFileName, "wb") as pickleFile:
					pickle.dump(advSet, pickleFile, protocol=pickle.HIGHEST_PROTOCOL)

			print ("Train adversarial examples: %d | Test adversarial examples: %d" % (np.sum(advSet == 0) / 2, np.sum(advSet == 1) / 2))

	print ("Adversarial examples generated successfully!")

import os
import sys
import cv2
import pprint
import numpy as np

try:
	import cPickle as pickle # Python2
except ModuleNotFoundError:
	import pickle # Python3
	
if len(sys.argv) < 4:
	print ("Usage: python plotAdvImages.py <Directory> <Old File Format?> <Display Images?>")
	exit (-1)

directoryName = sys.argv[1]
oldFileFormat = int(sys.argv[2]) # True or False
displayImages = int(sys.argv[3]) # True or False
advExamplesClass = 14 # Stop sign class
setDict = {0: "Train", 1: "Test"}

# Iterate over the directories
numAdvExamples = {}
for root, dirs, files in os.walk(directoryName):
	path = root.split('/')
	print ("Directory:", os.path.basename(root))

	directoryName = root
	imageFileName = os.path.join(directoryName, "Images.pickle")
	if not os.path.exists(imageFileName):
		continue
	if directoryName not in numAdvExamples:
		numAdvExamples[directoryName] = {}
		numAdvExamples[directoryName]["Train"] = 0
		numAdvExamples[directoryName]["Test"] = 0
	
	if oldFileFormat:
		print ("Loading files from directory: %s" % directoryName)
		with open(os.path.join(directoryName, "Images.pickle"), "rb") as pickleFile:
			advImages = pickle.load(pickleFile)
		with open(os.path.join(directoryName, "Labels.pickle"), "rb") as pickleFile:
			advLabels = pickle.load(pickleFile)
		with open(os.path.join(directoryName, "Set.pickle"), "rb") as pickleFile:
			advSet = pickle.load(pickleFile)
	else:
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

		pickleFileName = os.path.join(directoryName, directoryName.split(os.sep)[-1] + ".pickle")
		print ("Loading file: %s" % pickleFileName)
		with open(pickleFileName, "rb") as pickleFile:
			data = pickle.load(pickleFile)

		data.print_data_shape()

	numAdvExamples[directoryName]["Train"] = (np.sum(advSet == 0) / 2) if oldFileFormat else data._X_train.shape[0]
	numAdvExamples[directoryName]["Test"] = (np.sum(advSet == 1) / 2) if oldFileFormat else data._X_test.shape[0]

	for i in range(0, advLabels.shape[0] if oldFileFormat else data._X_test.shape[0], 2 if oldFileFormat else 1):
		if oldFileFormat:
			image = advImages[i, :, :, :]
			advImage = advImages[i+1, :, :, :]

		else:
			if data._y_test[i] != advExamplesClass:
				continue
			image = data._X_test[i, :, :, :]
			advImage = data._X_test_adv[i, :, :, :]

		diff = np.sum(image - advImage)
		if oldFileFormat:
			print ("Set: %s | Difference: %f" % (setDict[advSet[i]], diff))
		else:
			print ("Original image label: %d | Adversarial image label: %d | Difference: %f" % (data._y_test[i], data._y_test_adv[i], diff))
		
		if displayImages:
			cv2.imshow("Original Image", np.clip(image[:, :, :] * 255.0, 0.0, 255.0).astype(np.uint8))
			cv2.imshow("Adversarial Image", np.clip(advImage[:, :, :] * 255.0, 0.0, 255.0).astype(np.uint8))

			key = cv2.waitKey(-1)
			if key == ord('q'):
				print ("Process terminated by user!")
				exit (-1)
			if key == ord('b'):
				print ("Directory skipped!")
				break

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(numAdvExamples)

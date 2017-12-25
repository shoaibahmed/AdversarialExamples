import os
import shutil

ROOT_DIRECTORY = "/mnt/BoundaryAttack/data/ILSVRC2012_img_test"
IMAGE_FILE_EXTENSIONS = [".png", ".bmp", ".jpg", ".JPEG"]
outputFile = open(os.path.join(ROOT_DIRECTORY, "../filenames-complete.txt"), "w")

for root, dirs, files in os.walk(ROOT_DIRECTORY):
	# Iterate over the files
	for file in files:
		# Check if the file is an image
		isImage = False
		for ext in IMAGE_FILE_EXTENSIONS:
			if file.endswith(ext):
				isImage = True
				break

		if not isImage:
			continue

		# Write full image path to file
		absoluteFilePath = os.path.abspath((os.path.join(root, file)))
		outputFile.write("%s\n" % str(absoluteFilePath))

outputFile.close()
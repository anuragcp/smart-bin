import mvnc.mvncapi as mvnc
import cv2 as cv
import numpy
import skimage

GRAPH_PATH = "./smart_bin.graph"
IMAGE_PATH = "./img.jpg"
IMG_DIM = (1,4004)
IMG_MEAN = 0.5
IMG_STDDEV = 1
LABEL_FILE_PATH = "./retrained_labels.txt"
CAM_INDEX = 0

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
	print("No device found...")
	quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

with open(GRAPH_PATH, mode="rb") as f:
	blob = f.read()

graph = device.AllocateGraph(blob)

#img = skimage.io.imread(IMAGE_PATH)
cap = cv.VideoCapture(CAM_INDEX)
while (True):
	ret, img = cap.read()
	frame = img
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = skimage.transform.resize(img, IMG_DIM, preserve_range= True)
	img = img.astype(numpy.float32)
	img = (img - IMG_MEAN) * (IMG_STDDEV)
	graph.LoadTensor(img.astype(numpy.float16), "usre_object")
	output, userobj = graph.GetResult()
	labels = numpy.loadtxt(LABEL_FILE_PATH, str, delimiter = '\n')
	print("Output = {}".format(labels[output.argmax()]))
	cv.imshow('out', frame)
	if (cv.waitKey(1) % 0xFF == ord('q')):
		break;


graph.DeallocateGraph()
device.CloseDevice()

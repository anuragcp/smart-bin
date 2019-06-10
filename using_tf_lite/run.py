#import RPi.GPIO as GPIO
import time
from label_image import *
#from picamera import PiCamera
from PIL import Image
import numpy as np
import cv2
import time

img = []
model_file = "./model/retrained_graph.pb"
file_name = "./image/test.jpg"
label_file = "model/retrained_labels.txt"
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"
#if utrasonic detects a object
#stream = BytesIO()
#camera = PiCamera()
#camera.start_preview()
#sleep(2)
#camera.capture(stream, format='jpeg')
#stream.seek(0)

cap = cv2.VideoCapture(1)

#create a if function here for ultrasonic sensor readings
ret, frame = cap.read()
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#saving image with time span
cv2.imwrite('./temp/test.jpg',frame)
cv2.imwrite('/image/{}.png'.format(time.ctime(time.time())),frame)


img = Image.open("./image/test.jpg")
img = np.array(img)
print("Image resized to {}".format(299,299))

graph = load_graph(model_file)
print("Model loaded...")
print(img)
t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)
input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name);
output_operation = graph.get_operation_by_name(output_name);
with tf.Session(graph=graph) as sess:
	start = time.time()
	results = sess.run(output_operation.outputs[0],{input_operation.outputs[0]: t})
	end=time.time()
results = np.squeeze(results)
top_k = results.argsort()[-5:][::-1]
labels = load_labels(label_file)
print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
template = "{} (score={:0.5f})"
for i in top_k:
	print(template.format(labels[i], results[i]))


"""GPIO.setmode(GPIO.BOARD)
GPIO.setup(12, GPIO.OUT)
p = GPIO.PWM(12, 50)
p.start(7.5)

try:
    while True:
        p.ChangeDutyCycle(7.5)  # turn towards 90 degree
        time.sleep(1) # sleep 1 second
        p.ChangeDutyCycle(2.5)  # turn towards 0 degree
        time.sleep(1) # sleep 1 second
        p.ChangeDutyCycle(12.5) # turn towards 180 degree
        time.sleep(1) # sleep 1 second 
except KeyboardInterrupt:
    p.stop()
    GPIO.cleanup()"""

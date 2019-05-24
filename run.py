import RPi.GPIO as GPIO
import time
import label_image
from picamera import PiCamera
from PIL import Image

img = []
model_file = "./model/retrained_graph.pb"
#if utrasonic detects a object
stream = BytesIO()
camera = PiCamera()
camera.start_preview()
sleep(2)
camera.capture(stream, format='jpeg')
stream.seek(0)
img = Image.open(stream)

graph = load_graph(model_file)
t = read_tensor_from_image_file(file_name = img,
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
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
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
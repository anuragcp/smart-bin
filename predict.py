import numpy as np
import os
from scipy import  misc
from keras.models import model_from_json
import pickle
import cv2

bio = {0,1,5}
nonbio = {2,3,4,6,7,8}

#classifier_f = open("int_to_word_out.pickle", "rb")
#int_to_word_out = pickle.load(classifier_f)
#classifier_f.close()

def load_model():

    # load json and create model
    json_file = open('model_face.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_smartbin.h5")
    print("Loaded model from disk")
    return loaded_model

def pre_process(image):
    image = image.astype('float32')
    image = image / 255.0
    return image

def load_image():

    img=os.listdir("predict")[0]
    image=np.array(misc.imread("predict/"+img))
    image = misc.imresize(image, (224, 224))
    image=np.array([image])
    image=pre_process(image)
    return image

model=load_model()
"""
image=load_image()
prediction=model.predict(image)
print(prediction)
pred = np.argmax(prediction)
if pred in bio:
    print("material is bio")
else:
    print("material is non-bio")
"""

cap = cv2.VideoCapture(0)

while True:
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        key_condition = cv2.waitKey(1) & 0xFF
        if key_condition == ord('c'):
            cv2.imwrite("./predict/temp.jpg",frame)
            break
        elif key_condition == ord('q'):
            exit()
        else:
            pass

    image=load_image()
    prediction=model.predict(image)
    print(prediction)
    pred = np.argmax(prediction)
    if pred in bio:
        print("material is bio")
    else:
        print("material is non-bio")


#print(int_to_word_out[np.argmax(prediction)])

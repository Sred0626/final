import argparse
import os
import json
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")  # default assume that our export is in this file's parent directory


class Model(object):
    def __init__(self, model_dir=MODEL_DIR):
        # make sure our exported SavedModel folder exists
        model_path = os.path.realpath(model_dir)
        if not os.path.exists(model_path):
            raise ValueError(f"Exported model folder doesn't exist {model_dir}")
        self.model_path = model_path

        # load our signature json file, this shows us the model inputs and outputs
        # you should open this file and take a look at the inputs/outputs to see their data types, shapes, and names
        with open(os.path.join(model_path, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")

        # placeholder for the tensorflow session
        self.session = None

    def load(self):
        self.cleanup()
        # create a new tensorflow session
        self.session = tf.compat.v1.Session(graph=tf.Graph())
        # load our model into the session
        tf.compat.v1.saved_model.loader.load(sess=self.session, tags=self.signature.get("tags"), export_dir=self.model_path)

    def predict(self, image: Image.Image):
        # load the model if we don't have a session
        if self.session is None:
            self.load()
        # get the image width and height
        width, height = image.size
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
        # now the image is square, resize it to be the right shape for the model input
        if "Image" not in self.inputs:
            raise ValueError("Couldn't find Image in model inputs - please report issue to Lobe!")
        input_width, input_height = self.inputs["Image"]["shape"][1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))
        # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
        image = np.asarray(image) / 255.0
        # create the feed dictionary that is the input to the model
        # first, add our image to the dictionary (comes from our signature.json file)
        feed_dict = {self.inputs["Image"]["name"]: [image]}

        # list the outputs we want from the model -- these come from our signature.json file
        # since we are using dictionaries that could have different orders, make tuples of (key, name) to keep track for putting
        # the results back together in a dictionary
        fetches = [(key, output["name"]) for key, output in self.outputs.items()]

        # run the model! there will be as many outputs from session.run as you have in the fetches list
        outputs = self.session.run(fetches=[name for _, name in fetches], feed_dict=feed_dict)
        # do a bit of postprocessing
        results = {}
        # since we actually ran on a batch of size 1, index out the items from the returned numpy arrays
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        return results

    def cleanup(self):
        # close our tensorflow session if one exists
        if self.session is not None:
            self.session.close()
            self.session = None

    def __del__(self):
        self.cleanup()

cap = cv2.VideoCapture(0) 
#擷取畫面 寬度 設定為640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#擷取畫面 高度 設定為480
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

model = Model()
model.load()

while(1):
    # get a frame
    ret, frame = cap.read()

    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    outputs = model.predict(image)

    print(f"Predicted: {outputs}")
    #show detial on window
    frame = cv2.putText(frame,outputs['Prediction'] + " " +
        str(round(max(outputs['Confidences']),3)),
        (10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),1, cv2.LINE_AA)
    # show a frame
    cv2.imshow("Detcting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
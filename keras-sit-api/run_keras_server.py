# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
from threading import Thread
from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io
# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"
REDIS_HOST="redis"
# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
#db = redis.StrictRedis(host="localhost", port=6379, db=0)
db = redis.StrictRedis(host=REDIS_HOST)
model = None

#Redis will act as our temporary data store on the server. Images will come in to the server via a variety of methods such as cURL, a Python script, or even a mobile app.
#Furthermore, images could come in only every once in awhile (a few every hours or days) or at a very high rate (multiple per second). We need to put the images somewhere as they queue up prior to being processed. Our Redis store will act as the temporary storage.
#In order to store our images in Redis, they need to be serialized. Since images are just NumPy arrays, we can utilize base64 encoding to serialize the images. Using base64 encoding also has the added benefit of allowing us to use JSON to store additional attributes with the image.
#Our base64_encode_image function handles the serialization.

def base64_encode_image(a):
	# base64 encode the input NumPy array
	return base64.b64encode(a).decode("utf-8")

#Similarly, we need to deserialize our image prior to passing them through our model. This is handled by the base64_decode_image function

def base64_decode_image(a, dtype, shape):
	# if this is Python 3, we need the extra step of encoding the
	# serialized NumPy string as a byte object
	if sys.version_info.major == 3:
		a = bytes(a, encoding="utf-8")
	# convert the string to a NumPy array using the supplied data
	# type and target shape
	a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
	a = a.reshape(shape)
	# return the decoded image
	return a

#prepare_image function which pre-processes our input image for classification using the ResNet50 implementation in Keras.
#When utilizing your own models I would suggest modifying this function to perform any required pre-processing, scaling, or normalization.

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	# return the processed image
	return image

#The classify_process function will be kicked off in its own thread as we’ll see in __main__ below. 
#Loading the model happens only once when this thread is launched — it would be terribly slow 
#if we had to load the model each time we wanted to process an image and furthermore it could lead to a server crash due to memory exhaustion.
#This function will poll for image batches from the Redis server, classify the images, and return the results to the client.

def classify_process():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	print("* Loading model...")
	model = ResNet50(weights="imagenet")
	print("* Model loaded")
	# continually pool for new images to classify
	while True:
		# attempt to grab a batch of images from the database, then
		# initialize the image IDs and batch of images themselves
		queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        #Here we’re first using the Redis database’s lrange function to get, at most, BATCH_SIZE images from our queue
		imageIDs = []
		batch = None
		# loop over the queue
		for q in queue:
			# deserialize the object and obtain the input image
			q = json.loads(q.decode("utf-8"))
			image = base64_decode_image(q["image"], IMAGE_DTYPE,
				(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
			# check to see if the batch list is None
			if batch is None:
				batch = image
			# otherwise, stack the data
			else:
				batch = np.vstack([batch, image])
			# update the list of image IDs
			imageIDs.append(q["id"])

		# check to see if we need to process the batch
		if len(imageIDs) > 0:
			# classify the batch
			print("* Batch size: {}".format(batch.shape))
			preds = model.predict(batch) #we make predictions on the entire batch by passing it through the model
			results = decode_predictions(preds)
			# loop over the image IDs and their corresponding set of
			# results from our model to append labels and probabilities to an output list 
            # and then store the output in the Redis database using the imageID as the key
			for (imageID, resultSet) in zip(imageIDs, results):
				# initialize the list of output predictions
				output = []
				# loop over the results and add them to the list of
				# output predictions
				for (imagenetID, label, prob) in resultSet:
					r = {"label": label, "probability": float(prob)}
					output.append(r)
				# store the output predictions in the database, using
				# the image ID as the key so we can fetch the results
				db.set(imageID, json.dumps(output))
			# remove the set of images that we just classified from our queue using ltrim
			db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

		# sleep for a small amount
		time.sleep(SERVER_SLEEP)

@app.get("/")
def index():
    return "Say Hello to your AI Overlord!"

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):

			# read the image in PIL format and prepare it for classification
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")
			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			d = {"id": k, "image": base64_encode_image(image)}
			db.rpush(IMAGE_QUEUE, json.dumps(d))    

			# keep looping until our model server returns the output predictions
			while True:
				# attempt to grab the output predictions
				output = db.get(k)
				# check to see if our model has classified the input
				# image
				if output is not None:
 					# add the output predictions to our data
 					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)
					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break
				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(CLIENT_SLEEP)
			# indicate that the request was a success
			data["success"] = True
	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	# load the function used to classify input images in a *separate*
	# thread than the one used for main classification
	print("* Starting model service...")
	t = Thread(target=classify_process, args=())
	t.daemon = True
	t.start()
	# start the web server
	print("* Starting web service...")
	app.run()

'''
However, there is a subtle problem…
Depending on how you deploy your deep learning REST API, there is a subtle problem with keeping the classify_process function in the same file as the rest of our web API code.

Most web servers, including Apache and nginx, allow for multiple client threads.

If you keep classify_process in the same file as your predict view, then you may load multiple models if your server software deems it necessary to create a new thread to serve the incoming client requests — for every new thread, a new view will be created, and therefore a new model will be loaded.

The solution is to move classify_process to an entirely separate process and then start it along with your Flask web server and Redis server.

In next week’s blog post I’ll build on today’s solution, show how to resolve this problem, and demonstrate:

How to configure the Apache web server to serve our deep learning REST API
How to run classify_process as an entirely separate Python script, avoiding “multiple model syndrome”
Provide stress test results, confirming and verifying that our deep learning REST API can scale under heavy load
'''	
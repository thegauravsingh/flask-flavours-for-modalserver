# Here in run_web_server.py , you’ll see predict , the function associated with our REST API /predict endpoint.
# The predict function pushes the encoded image into the Redis queue and then continually loops/polls 
# until it obains the prediction data back from the model server. 
# We then JSON-encode the data and instruct Flask to send the data back to the client.

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import settings as settings
import helpers as helpers
from fastapi import FastAPI, File, HTTPException
from starlette.requests import Request
import redis
import uuid
import time
import json
import io
# initialize our Flask application and Redis server
app = FastAPI()
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)
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

@app.get("/")
def homepage():
	return "Welcome to the Keras Fast API!"

@app.post("/predict")
def predict(request: Request, img_file: bytes=File(...)):
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	# ensure an image was properly uploaded to our endpoint
	if request.method == "POST":
		# read the image in PIL format and prepare it for
		# classification
		image = Image.open(io.BytesIO(img_file))
		image = prepare_image(image,
			(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
		# ensure our NumPy array is C-contiguous as well,
		# otherwise we won't be able to serialize it
		image = image.copy(order="C")
		# generate an ID for the classification then add the
		# classification ID + image to the queue
		k = str(uuid.uuid4())
		image = helpers.base64_encode_image(image)
		d = {"id": k, "image": image}
		db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
		# keep looping until our model server returns the output
		# predictions
        # Keep looping for CLIENT_MAX_TRIES times
		num_tries = 0
		while num_tries <= settings.CLIENT_MAX_TRIES:
			
			num_tries += 1
			# Attempt to grab the output predictions
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
			time.sleep(settings.CLIENT_SLEEP)
			# indicate that the request was a success
			data["success"] = True
		else:
			raise HTTPException(status_code=400, detail="Request failed after {} tries".format(settings.CLIENT_MAX_TRIES))

    # Return the data dictionary as a JSON response
	return data

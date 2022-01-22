'''
How to (and how not to) load a Keras model into memory so it can be efficiently used for inference
How to use the Flask web framework to create an endpoint for our API
How to make predictions using our model, JSON-ify them, and return the results to the client
How to call our Keras REST API using both cURL and Python

Building your Keras REST API
Our Keras REST API is self-contained in a single file named run_keras_server.py. We kept the installation in a single file as a manner of simplicity — the implementation can be easily modularized as well.

Inside run_keras_server.py you'll find three functions, namely:

load_model: Used to load our trained Keras model and prepare it for inference.
prepare_image: This function preprocesses an input image prior to passing it through our network for prediction. If you are not working with image data you may want to consider changing the name to a more generic prepare_datapoint and applying any scaling/normalization you may need.
predict: The actual endpoint of our API that will classify the incoming data from the request and return the results to the client.
'''

# import the necessary packages
from tensorflow.keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model

    # As the name suggests, this method is responsible for instantiating our architecture and loading our weights from disk.
    # For the sake of simplicity, we'll be utilizing the ResNet50 architecture which has been pre-trained on the ImageNet dataset.
    # If you're using your own custom model you'll want to modify this function to load your architecture + weights from disk.
    model = ResNet50(weights="imagenet") 

# Before we can perform prediction on any data coming from our client we first need to prepare and preprocess the data:
def prepare_image(image, target):
 
    #This function:
    #   Accepts an input image
    #   Converts the mode to RGB (if necessary)
    #   Resizes it to 224x224 pixels (the input spatial dimensions for ResNet)
    #   Preprocesses the array via mean subtraction and scaling
    #   Again, you should modify this function based on any preprocessing, scaling, and/or normalization you need prior to passing the input data through the model.

    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

#We are now ready to define the predict function — this method processes any requests to the /predict endpoint:
@app.route("/predict", methods=["POST"])
def predict():
    
    # The data dictionary is used to store any data that we want to return to the client. 
    # Right now this includes a boolean used to indicate if prediction was successful or not 
    # we'll also use this dictionary to store the results of any predictions we make on the incoming data.
    
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    # To accept the incoming data we check if:
    #   The request method is POST (enabling us to send arbitrary data to the endpoint, including images, JSON, encoded-data, etc.)
    #   An image has been passed into the files attribute during the POST
    
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # If you're working with non-image data you should remove the request.files code and either parse the raw input data yourself 
        # or utilize request.get_json() to automatically parse the input data to a Python dictionary/object. 
        # Additionally, consider giving 'https://www.digitalocean.com/community/tutorials/processing-incoming-request-data-in-flask' 
        # tutorial a read which discusses the fundamentals of Flask's request object.
        if flask.request.files.get("image"):
            
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image 
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            
            # initialize the list of predictions to return to the client
            data["predictions"] = []

            # loop over the results and add them to the list of returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and then start the server
# so let's launch our service:
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()
# First we call load_model which loads our Keras model from disk.
# The call to load_model is a blocking operation and prevents the web service from starting until the model is fully loaded. 
# Had we not ensured the model is fully loaded into memory and ready for inference prior to starting the web service we could run into a situation where:
# A request is POST'ed to the server.
# The server accepts the request, preprocesses the data, and then attempts to pass it into the model
# ...but since the model isn't fully loaded yet, our script will error out!
# Lesson: When building your own Keras REST APIs, ensure logic is inserted to guarantee your model is loaded and ready for inference prior to accepting requests.    
# Reason: If You are tempted to load your model inside your predict function, implies that the model will be loaded each and every time a new request comes in. 
# This is incredibly inefficient and can even cause your system to run out of memory as you'll notice that your API will run considerably slower 
# due to the significant overhead in both I/O and CPU operations used to load your model for each new request.

#Caveat: assuming you are using the default Flask server that is single threaded. 
#If you deploy to a multi-threaded server you could be in a situation where you are still loading multiple models in memory even when using the "more correct" method discussed earlier in this post. 
#If you intend on using a dedicated server such as Apache or nginx you should consider making your pipeline more scalable

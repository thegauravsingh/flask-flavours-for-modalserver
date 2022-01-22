# The helpers.py file contains two functions — one for base64 encoding and the other for decoding.
# Encoding is necessary so that we can serialize + store our image in Redis. 
# Likewise, decoding is necessary so that we can deserialize the image into NumPy array format prior to pre-processing.

# import the necessary packages
import numpy as np
import base64
import sys
def base64_encode_image(a):
	# base64 encode the input NumPy array
	return base64.b64encode(a).decode("utf-8")
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
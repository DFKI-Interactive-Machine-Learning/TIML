import sys

import keras

from timl.classification.classifier import Classifier

"""
E.g.:
python ClassifyImage.py ../../GoodModels/20190330-130503-keras_model-5.h5 ../../../DataSets/ISIC/ISIC-190110/Images/ISIC_0000000.jpeg
"""

if len(sys.argv) < 3:
    print("Usage: ClassifyImage <modelpath:str> <imagepath:str>")
    exit(10)

model_filepath = sys.argv[1]
image_filepath = sys.argv[2]


print("Loading Keras model from {} ...".format(model_filepath))
model = keras.models.load_model(filepath=model_filepath)
print("Model Loaded.")

print("Classifying image {} ...".format(image_filepath))

predicted_prob_distribution = Classifier.classify(model=model, image_path=image_filepath)

print("Prediction: {}".format(predicted_prob_distribution))

print("Done.")

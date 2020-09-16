from keras.models import Model, load_model
import numpy as np
import pandas as pd
import os
import argparse

from timl.classification.classifier import Classifier
from timl.common.imageaugmentation import image_provider_factory
from timl.networking.config import get_isic_base_path

ISIC_DATASET_IMG_DIR = get_isic_base_path()


#
# Arguments
args_parser = argparse.ArgumentParser()
args_parser.add_argument('model', metavar='<keras_model.h5>', type=str,
                         help="Path to the h5 model")
args_parser.add_argument('layers', metavar='<layer_name>', type=str,
                         help="The semi-column separated name of the layers you want to extract info from.")
args_parser.add_argument('augmentation', metavar='<augmentation_preset>', type=str,
                         help="Augmentation preset ('none', 'hflip', 'hflip_rot24', ....")
args_parser.add_argument('images_dataframe', metavar='<images.csv>', type=str,
                         help="Path to the csv file containing all the image names in column 'image_name'")
args_parser.add_argument('save_dir', metavar='<save_dir>', type=str,
                         help="Path to save activations to. Will create one sub-directrory for each layer.")
args_parser.add_argument('--img-dir', dest='img_dir', type=str,
                         help='The directory path in which to look for the images.'
                              ' If omitted, uses the one specified in skincare_config.json')
args_parser.add_argument('--cuda-gpu', dest='cuda_gpu', type=int,
                         help='The CUDA GPU number to use for training')

args = args_parser.parse_args()


class ActivationExtractor:

    def __init__(self, model: Model, nodes_to_evaluate: list):
        import keras.backend as K

        self.model = model
        self.nodes_to_evaluate = nodes_to_evaluate

        if not model._is_compiled:
            print('Please compile your model first! https://keras.io/models/model/#compile.')
            print('If you only care about the activations (outputs of the layers), '
                  'then just compile your model like that:')
            print('model.compile(loss="mse", optimizer="adam")')
            raise Exception('Compilation of the model required.')

        self.symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        self.f = K.function(self.symb_inputs, nodes_to_evaluate)



    # From: https://github.com/philipperemy/keract/blob/master/keract/keract.py
    def evaluate(self, x, y=None):
        """
        Example:
        l1 = layer_outputs = model.layer[15].output  # something in between
        l2 = layer_outputs = model.layer[-1].output  # the output
        activations = _evaluate(model, layer_outputs=[l1, l2], x, y=None)

        assert len(activations) == 2
        assert len(activations[0]) == size of layer l1

        """

        x_, y_, sample_weight_ = self.model._standardize_user_data(x, y)
        return self.f(x_ + y_ + sample_weight_)

#
# Extract arguments
path_model = args.model
layer_names = args.layers.split(';')
augmentation_preset = args.augmentation
images_csv = args.images_dataframe
save_dir = args.save_dir

if not os.path.exists(save_dir):
    raise Exception("Output directory {} doesn't exists".format(save_dir))

if args.cuda_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_gpu)

# Overrides the default images dir
if args.img_dir is not None:
    ISIC_DATASET_IMG_DIR = args.img_dir

#
# Compose the model that will read information from the specified layer
print("Loading '{}'".format(path_model))
mmodel = load_model(path_model)  # type: Model
print("==================================")
print("Model structure:")
print(mmodel.summary())
print("==================================")
extraction_model = Model(inputs=mmodel.input, outputs=mmodel.get_layer(layer_names[0]).output)
print("Extraction model structure:")
print(extraction_model.summary())
print("==================================")

#
# Check for the existence of the layers
print("Checking fo layers ", layer_names)
all_layer_names = [l.name for l in mmodel.layers]
for l in layer_names:
    if l not in all_layer_names:
        raise Exception("Couldn't find layer {} in the model.".format(l))

#
# Read the images csv
print("Loading '{}'".format(images_csv))
images_df = pd.read_csv(images_csv)
N = len(images_df)
print("Size of dataset: {} images".format(N))
image_names = images_df['image_name']
assert len(image_names) == N

#
# Find the image size that the model takes as input
l0 = extraction_model.get_layer(index=0)
# Get the dimension (the first number is the sample number)
_, w, h, depth = l0.input_shape
img_size = (w, h)
print("Using image size {}".format(img_size))

#
#
print("Checking images...")
image_paths = []
for i, img_name in enumerate(image_names):
    # print("{} - {}".format(i, img_name))
    img_path = os.path.join(ISIC_DATASET_IMG_DIR, img_name + ".jpg")
    if not os.path.exists(img_path):
        raise Exception("Could not find image path '{}'".format(img_path))
    image_paths.append(img_path)

# Do not resize the image. It will already performed by the `classify` method.
img_provider = image_provider_factory(config=augmentation_preset, image_paths=image_paths,
                                      resize=None, resize_filter=None, color_space='RGB')

# Number of augmented images
Naug = img_provider.num_images()

aug_factor = int(Naug / N)
print("Augmentation factor: {}".format(aug_factor))

#
# Preparing the list of layer outputs and directories
layer_outputs = []
layer_dirs = []
for lname in layer_names:
    layer = mmodel.get_layer(lname)
    layer_outputs.append(layer.output)

    layer_dir = os.path.join(save_dir, lname)
    print("Creating directory ", layer_dir)
    if not os.path.exists(layer_dir):
        os.mkdir(layer_dir)
    layer_dirs.append(layer_dir)

#
# predict
print("Extracting...")

act_extractor = ActivationExtractor(model=mmodel, nodes_to_evaluate=layer_outputs)

i_aug = 0
for i_orig in range(N):
    for aug in range(aug_factor):

        img_name = image_names[i_orig]
        filename = "{}-{:010d}.npy".format(img_name, aug)
        print("idx {} - {}".format(i_aug, filename), end="")

        # Old way to get the activations, by using a wrapping model
        # and getting the output of a single layer at a time.
        # pred = Classifier.classify_image(extraction_model, img)
        # print("predicted data is {}".format(pred))

        img = img_provider.get_image(i_aug)
        img_np = Classifier.img_to_numpy_sample(image=img, model=mmodel)
        # activations = _evaluate(mmodel, layer_outputs, x=img_np)
        activations = act_extractor.evaluate(x=img_np)
        assert len(activations) == len(layer_outputs)

        for layer_num, layer_activations in enumerate(activations):
            print("...", layer_names[layer_num], end="")
            layer_dir = layer_dirs[layer_num]

            # Since we provided only 1 image
            assert layer_activations.shape[0] == 1
            layer_activations = layer_activations[0]

            # Force to 32 bits. 64 is too long.
            pred_np = np.array(layer_activations, dtype=np.float32)

            pred_file = os.path.join(layer_dir, filename)
            np.save(pred_file, pred_np)

        i_aug += 1  # next augmented index
        print(".")

print("All done.")

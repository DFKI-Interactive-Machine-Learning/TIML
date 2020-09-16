import os
import argparse

from keras.models import load_model
from PIL import Image

from timl.classification.classifier import Classifier


parser = argparse.ArgumentParser(
    description="Given an image (e.g., img.png), generates the XAI (eXplainable AI) images:"
                " i) the saliency map (as img-greymap.png)"
                " ii) the heat map (as img-heatmap.png)"
                " iii) and the composition between the original image and the heatmap (as img-composite.png)."
                " The resolution of the images refers to the input resolution of the prediction model.")
parser.add_argument('image_path', metavar="<image.png/jpg>", type=str,
                    help='Address of image')
parser.add_argument('model_path', metavar="<keras_model.h5>", type=str,
                    help='Address of the model')
parser.add_argument('method', metavar="<method>", type=str,
                    help="The method to use for extraction, among: 'gradcam', 'rise'")
parser.add_argument('--gradcam-layer-name', type=str,
                    help='For method gradcam: the name of the layer, e.g block5_conv3')
parser.add_argument('--rise-iterations', type=int,
                    help='For method rise: the number of iterations (Suggested: between 1000 and 2000, default: 2000)')
parser.add_argument('--rise-mask-size', type=int,
                    help='For method rise: the size of the mask (suggested, between 2 and 8, default 6)')
parser.add_argument('--out-dir', type=str,
                    help='Optional output directory')
parser.add_argument('--cuda-gpu', type=str,
                    help='The number of the GPU that will process the keras model.')

args = parser.parse_args()

#
# Set CUDA GPU (do it before loading a model)
if args.cuda_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpu

out_dir = None
if args.out_dir is not None:
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        # raise Exception("Output directory '{}' doesn't exist.".format(out_dir))
        print("Creating directory '{}'...".format(out_dir))
        os.makedirs(out_dir)

#
# Load Keras model
if not os.path.exists(args.model_path):
    raise Exception("Model file {} doesn't exist".format(args.model_path))

print("Loading Keras model '{}'...".format(args.model_path))
model = load_model(args.model_path)
print("Available layers:")
for i, layer in enumerate(model.layers):
    print("{} - {}".format(i, layer.name))

#
# Select the method
xai_method = args.method
print("Using method '{}'".format(xai_method))


#
# Parse the optional parameters
method_params = dict()

if args.gradcam_layer_name is not None:
    method_params['layer_name'] = args.gradcam_layer_name

if args.rise_iterations is not None:
    method_params['N'] = args.rise_iterations

if args.rise_mask_size is not None:
    method_params['s'] = args.rise_mask_size

#
# Load the image
print("Loading Image...")

if not os.path.exists(args.image_path):
    raise Exception("Image file doesn't exist".format(args.image_path))
image_path = args.image_path
image = Image.open(image_path)

# Get Heatmap

print("Getting Heatmap...")
import sys
sys.path.append("/opt/nhmduy_research/SkinCareProject/timl/Classifiers/")
print("Getting maps...")


cls = Classifier()
cls._model = model
grey_map, heat_map, composite = cls.generate_heatmap(image=image, method=xai_method, **method_params)

# Save image
print("Saving maps...")
image_root_path, _ = os.path.splitext(image_path)

if out_dir is not None:
    _, img_rootname = os.path.split(image_root_path)
    image_root_path = os.path.join(out_dir, img_rootname)

grey_map.save(image_root_path + "-greymap.png")
heat_map.save(image_root_path + "-heatmap.png")
print (image_root_path + "-heatmap_overlay.png")
composite.save(image_root_path + "-heatmap_overlay.png")

print("All done.")

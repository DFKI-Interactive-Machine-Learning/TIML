import argparse


from keras.models import load_model
from keras.utils import plot_model

args_parser = argparse.ArgumentParser(
    description='Plotting a keras model.')

args_parser.add_argument('keras_model', metavar='<keras_model.h5>', type=str,
                         help="The keras model to use for plotting.")
args_parser.add_argument('outfile', metavar='<outfile.png>', type=str,
                         help="The name of the output image file.")
args_parser.add_argument('--lr', action="store_true",
                         help="If set, the plot will go form left to right, rather than from top to down.")

# This possibly stops the execution if the arguments are not correct
args = args_parser.parse_args()

model_filename = args.keras_model  # model full filename with path

#
# load the model
print("Loading the model")
model = load_model(model_filename)

#
#
print("Plotting model...")
# Use rankdir='TB' for top-down plots
rankdir = 'LR' if args.lr is True else 'TB'
plot_model(model=model, to_file=args.outfile, show_shapes=True, show_layer_names=True, rankdir=rankdir)

print("All done.")

import argparse
import os
import numpy as np
from timl.networking.config import get_isic_base_path
from keras.applications.vgg16 import (
    VGG16)
from timl.xai.gradcam.grad_cam_heat_map import Gradcam
#import pickle
import cv2
#from timl.common.datageneration import SkincareDataGenerator

ISIC_DATASET_IMG_DIR = get_isic_base_path()
if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(
        description='Automated testing of unknown classification')
    # args_parser.add_argument('test_image_data', metavar='<predict_images_df.csv>',
    #                          type=str, help='A csv file that contain image names and ground truth in one-hot format '
    #                                         'and must have \'UNK\' column in the .csv file')
    args_parser.add_argument('img_dir', metavar='<image-dir>', type=str,
                             help='A directory contain images for training')
    args_parser.add_argument('model', metavar='<model_information.csv>',
                             type=str, help='A csv file  congaing model information. Must have '
                                            'following columns {model_name: SVM,'
                                            ' svm_model: oneclass_svm.pickle, ........}')
    args_parser.add_argument('output_dir', metavar='<output_dir>', type=str,
                             help='A directory which will contain results and evaluation data ')
    args_parser.add_argument('--cuda-gpu', dest='cuda_gpu', type=int,
                             help='GPU number should be used for feature extraction from image ')

    args = args_parser.parse_args()
    if args.cuda_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_gpu)

        # Overrides the default images dir
    if args.img_dir is not None:
        ISIC_DATASET_IMG_DIR = args.img_dir


model = VGG16(weights='imagenet') # load_model(args.model)
heatmapgenerator = Gradcam(model)
print(args.img_dir)
preprocessed_input = heatmapgenerator.load_image(args.img_dir)


predictions = model.predict(preprocessed_input)

#top_1 = decode_predictions(predictions)[0][0]
print('Predicted class:')
#print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

predicted_class = np.argmax(predictions)
cam, heatmap = heatmapgenerator.grad_cam(model, preprocessed_input, predicted_class, 'block5_conv3')
cv2.imwrite(os.path.join(args.output_dir,"gradcam.jpg"), cam)

heatmapgenerator.register_gradient()
guided_model = heatmapgenerator.modify_backprop(model,args.model, 'GuidedBackProp')
saliency_fn = heatmapgenerator.compile_saliency_function(guided_model)
saliency = saliency_fn([preprocessed_input, 0])
gradcam = saliency[0] * heatmap[..., np.newaxis]
cv2.imwrite(os.path.join(args.output_dir, "guided_gradcam.jpg"), heatmapgenerator.deprocess_image(gradcam))
from timl.common.imageaugmentation import *
import numpy as np

import pkg_resources

# Compose the path of images, using the test data included in the `timl.data` module.
IMAGE_NAMES = ["ISIC_0000000.jpeg", "ISIC_0000000.jpeg"]
IMAGE_PATHS = [pkg_resources.resource_filename("timl.data", "sample_images/" + image_name) for image_name in IMAGE_NAMES]
IMAGE_SIZE = (450, 450)

#
# Configure the augmentation chain
#

# First instantiate the basic image provider, loading the images from disk
disk_image_provider = MultipleImageProvider(
    image_paths=IMAGE_PATHS,
    size=IMAGE_SIZE,
    resample_filter=PIL.Image.BILINEAR,
    color_space="RGB")

# Concatenate the basic provider with an augmenter performing H-Flip and another one performing 4x rotations
augmenter = RotatedImageAugmenter(rot_steps=4,
                                  provider=HFlipImageAugmenter(
                                      provider=disk_image_provider))

# Ask the augmenter the expected number of the augmented images
n_aug_images = augmenter.num_images()

# They should respect an 8x augmentation factor
assert n_aug_images == len(IMAGE_PATHS) * 8


#
# Cycle through the images
#
for i in range(n_aug_images):
    print("Getting image " + str(i))
    img = augmenter.get_image(i)

    # Verify that the size is still the same
    assert img.size == IMAGE_SIZE
    # Verify that the mode is indeed RGB (no alpha)
    assert img.mode == "RGB"
    # Convert into numpy format and print out the array shape
    img_np = np.asarray(img)
    print("Shape:", img_np.shape)


print("Done.")

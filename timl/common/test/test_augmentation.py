import os

import pytest
import PIL.Image
from PIL.Image import Image
from typing import List

import pkg_resources

from ..imageaugmentation import SingleImageProvider, HFlipImageAugmenter, RotatedImageAugmenter,\
    MultipleImageProvider, CachingImageAugmenter


TEST_IMG_NAMES = ["ISIC_0000000.jpg", "ISIC_0000001.jpg", "ISIC_0000002.jpg",
                  "ISIC_0000003.jpg", "ISIC_0000004.jpg"]


@pytest.fixture
def sample_image() -> Image:
    image_path = pkg_resources.resource_filename("timl.data", "sample_images/ISIC_0000000.jpeg")
    img = PIL.Image.open(image_path)

    return img


@pytest.fixture
def sample_images() -> List[Image]:
    out = []

    sample_images = pkg_resources.resource_listdir("timl.data", "sample_images")

    for img_filename in sample_images:
        if img_filename.startswith("."):
            continue

        image_path = pkg_resources.resource_filename("timl.data", "sample_images/" + img_filename)
        img = PIL.Image.open(image_path)
        out.append(img)

    return out


@pytest.fixture
def sample_images_path() -> List[str]:
    out = []

    sample_images = pkg_resources.resource_listdir("timl.data", "sample_images")

    for img_filename in sample_images:
        if img_filename.startswith("."):
            continue

        out.append(pkg_resources.resource_filename("timl.data", "sample_images/" + img_filename))

    return out


def test_imagespresence(sample_images_path):
    """Simply test if all sample images are really in the path"""
    for img_filename in sample_images_path:
        os.path.exists(img_filename)


def test_image_encoding_decoding(sample_image: Image, tmp_path):
    import numpy as np

    print("Testing img encodig / decoding in {}".format(tmp_path))

    assert sample_image.mode == "RGB"

    r_range, g_range, b_range = sample_image.getextrema()
    for pix_min, pix_max in [r_range, g_range, b_range]:
        assert pix_min >= 0.
        assert pix_max <= 255.0
        assert pix_min <= pix_max

    # Save the image as it is
    sample_image.save(os.path.join(tmp_path, "Original.png"))
    sample_image.save(os.path.join(tmp_path, "Original.jpg"))

    # Scale and save
    scaled_image = sample_image.resize((227, 227))
    scaled_image.save(os.path.join(tmp_path, "Scaled-227x227.png"))
    scaled_image.save(os.path.join(tmp_path, "Scaled-227x227.jpg"))

    # Encode in numpy format and save
    np_data = np.array(scaled_image, dtype=np.float32)
    # Normalize!
    np_data /= 255.0
    np.save(os.path.join(tmp_path, "Scaled-227x227.npy"), np_data)

    # Load back
    np_data_loaded = np.load(os.path.join(tmp_path, "Scaled-227x227.npy"))
    # Un-normalize!
    np_data_loaded *= 255.0

    # Back to Image
    np_data_bytes = np_data_loaded.astype(dtype=np.uint8)
    restore_img = PIL.Image.fromarray(np_data_bytes)
    restore_img.save(os.path.join(tmp_path, "Restored.png"))
    restore_img.save(os.path.join(tmp_path, "Restored.jpg"))

    # Assumptions
    assert sample_image.mode == restore_img.mode
    assert scaled_image.size == restore_img.size


def test_simple_save(tmp_path, sample_image):

    aug_simple = SingleImageProvider(sample_image)
    savename = os.path.join(tmp_path, "saved_img.png")
    print("Saving to " + savename)
    aug_simple.get_image(0).save(savename)


def test_imagechannels(sample_images):
    """Test if all test images are RGB"""
    for img in sample_images:  # type: Image
        assert len(img.getbands()) == 3


def test_imageflipper(sample_image):

    aug_simple = SingleImageProvider(sample_image)
    aug_flip = HFlipImageAugmenter(aug_simple)

    assert aug_simple.num_images() == 1
    assert aug_flip.num_images() == 2

    for i in range(aug_flip.num_images()):
        img = aug_flip.get_image(i)
        assert img is not None


def test_image_rotator(sample_images, tmp_path):

    steps = 6

    print("Saving rotated images to {}".format(tmp_path))

    img_id = 0
    for original_img in sample_images:  # type: Image
        aug_simple = SingleImageProvider(original_img)
        aug_rot = RotatedImageAugmenter(provider=aug_simple, rot_steps=steps)

        assert aug_simple.num_images() == 1
        assert aug_rot.num_images() == steps

        for i in range(aug_rot.num_images()):
            img = aug_rot.get_image(i)
            assert img is not None
            img_savepath = tmp_path / "img-{}.png".format(img_id)
            img.save(img_savepath)

            img_id += 1


def test_multiple_generators(tmp_path, sample_images_path):

    n_images = len(sample_images_path)

    print("Saving {} augmented images to {}".format(n_images, tmp_path))

    img_loader = MultipleImageProvider(image_paths=sample_images_path, size=(227, 227), resample_filter=PIL.Image.NEAREST, color_space='RGB')
    assert img_loader.num_images() == n_images

    rot_steps = 6

    aug_hflip = HFlipImageAugmenter(provider=img_loader)
    assert aug_hflip.num_images() == img_loader.num_images() * 2
    aug_rot_hflip = RotatedImageAugmenter(provider=aug_hflip, rot_steps=rot_steps)
    assert aug_rot_hflip.num_images() == aug_hflip.num_images() * rot_steps

    n_augmented_images = aug_rot_hflip.num_images()

    for i in range(n_augmented_images):
        augmented_img = aug_rot_hflip.get_image(i)
        img_savepath = tmp_path / "img-{}.png".format(i)
        augmented_img.save(img_savepath)


def test_image_caching(sample_images_path):

    n_images = len(sample_images_path)

    img_provider = MultipleImageProvider(image_paths=sample_images_path, size=None, resample_filter=PIL.Image.NEAREST, color_space='RGB')
    cacher = CachingImageAugmenter(provider=img_provider, cache_limit=n_images)

    n_requests = n_images * 3

    for i in range(n_requests):
        image_num = i % n_images
        _ = cacher.get_image(image_num)

    # Now the number of requests must be exactly the number of calls
    cache_stats = cacher.get_stats()
    assert cache_stats.requests == n_requests
    # The number of hits must be the total number of calls minus the initial round, when the cache was empty
    assert cache_stats.hits == n_requests - n_images

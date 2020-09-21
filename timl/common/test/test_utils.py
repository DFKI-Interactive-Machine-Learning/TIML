import pytest

import os

from PIL.Image import Image

from .test_augmentation import sample_images


def test_jaccard_similarity_index(sample_images):
    from ..utils import jaccard_similarity

    assert len(sample_images) >= 2

    # Consider two different images
    im1 = sample_images[0]  # type: Image
    im2 = sample_images[1]  # type: Image

    # resulting index must lay between 0 and 1
    j1 = jaccard_similarity(im1, im2)
    assert 0 <= j1 <= 1

    # the similarity with itself should be 1
    j_same1 = jaccard_similarity(im1, im1)
    assert j_same1 == 1.0

    #
    # Test on 1-channel images
    im_lum1 = im1.convert(mode='L')
    im_lum2 = im2.convert(mode='L')
    j2 = jaccard_similarity(im_lum1, im_lum2)
    assert 0 <= j2 <= 1
    j_same2 = jaccard_similarity(im_lum1, im_lum1)
    assert j_same2 == 1


def test_circular_mask_creation(tmp_path):
    from ..utils import generate_circular_mask
    from ..utils import generate_circular_mask_bresenham

    from random import randint

    print("Testing circular mask creation in {}".format(tmp_path))

    #
    # test with several random sizes
    for n in range(10):
        w = randint(10, 800)
        h = randint(10, 600)

        m = generate_circular_mask(size=(w, h))
        m.save(os.path.join(tmp_path, "circular_mask-{}-{}x{}.png".format(n, w, h)))

        mb = generate_circular_mask_bresenham(size=(w, h))
        mb.save(os.path.join(tmp_path, "circular_mask_bresenham-{}-{}x{}.png".format(n, w, h)))


def test_random_image_creation(tmp_path):

    from ..utils import generate_random_image

    from random import randint

    print("Testing random image creation in {}".format(tmp_path))

    for n in range(10):
        w = randint(10, 800)
        h = randint(10, 600)

        img = generate_random_image((w, h))
        img.save(os.path.join(tmp_path, "random_image-{}x{}.png".format(w, h)))

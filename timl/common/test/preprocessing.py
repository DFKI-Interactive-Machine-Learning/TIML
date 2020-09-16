
import numpy as np
from PIL import Image


def extension(im2: Image, factor):
    dim2 = im2.size

    # extension factor
    new_dim = tuple(map(lambda x: int(x * factor), dim2))
    im2_ext = im2.resize(new_dim)
    a1_im = np.array(im2_ext)

    # enforce pixels to 0 and 255 after resizing
    a_im = np.where(a1_im < 128, 0, 255).astype(np.uint8)

    # get the un_scaled dimension of matrix
    unscaled_dim = tuple(map(lambda x: int(np.ceil(x / factor)), a_im.shape))

    # Get the difference between the enlarged image  and original image
    difference = tuple(map(lambda x, y: x - y, a_im.shape, unscaled_dim))

    # Get the higher and lower margins for the dimensions
    # rows
    l_ind1 = int(difference[0] / 2)
    h_ind1 = int(a_im.shape[0] - np.ceil(difference[0] / 2))

    # columns
    l_ind2 = int(difference[1] / 2)
    h_ind2 = int(a_im.shape[1] - np.ceil(difference[1] / 2))

    # Index the enlarged image using the dimensions calculated above
    a_ext = a_im[l_ind1:h_ind1, l_ind2:h_ind2]
    mask_ext = Image.fromarray(a_ext, mode='L')

    return mask_ext


def superimpose(im1, im2):
    # dim1: RGB image
    # dim2: the mask of dim1 - grayscale

    dim1, dim2 = im1.size, im2.size

    assert dim1 == dim2, "Dimension of image and mask must be the same"
    # rgb image without mask
    a_im1 = np.array(im1)

    # corresponding mask of the image
    a_im2 = np.array(im2)

    assert len(a_im2.shape) == 2, "image 2 should be mask in gray scale"

    # test that the image is in channel-last format
    dimension = list(a_im1.shape)
    dimension.remove(a_im1.shape[-1])
    d = list(map(lambda x: x > a_im1.shape[-1], dimension))
    assert all(d), "image should be in channel-last format"

    imp = np.zeros(shape=a_im1.shape, dtype='uint8')

    for i in range(a_im1.shape[2]):
        # across the channels
        imp[:, :, i] = np.where(a_im2 == 0, 0, a_im1[:, :, i])

    img_img = Image.fromarray(imp, mode='RGB')

    return img_img


def scan(imc: Image):
    n_dim = min(imc.size)

    # square dimension so that u can iterate rows and column in one loop
    imc_square = imc.resize((n_dim, n_dim))
    a_imc = np.array(imc_square)

    # contains the pixel value of a blob
    con = 255

    parity1 = [None, None]
    parity2 = [None, None]
    change = [0, 0]
    loc = []
    loc2 = []

    for i, j in zip(np.arange(a_imc.shape[1]), np.arange(a_imc.shape[0])):

        # gives True or False output
        T_FC = np.any(np.intersect1d(a_imc[:, i], con))  # 1 column at a time
        T_FR = np.any(np.intersect1d(a_imc[j, :], con))  # 1 row at a time

        if i == 0:
            parity1[0], parity1[1] = T_FC, T_FC
            parity2[0], parity2[1] = T_FR, T_FR

        # check if blob appears along the column
        parity1[0] = np.any(np.intersect1d(a_imc[:, i], con))
        # check if lesion mask appears along the row
        parity2[0] = np.any(np.intersect1d(a_imc[j, :], con))

        if parity1[0] != parity1[1]:
            change[0] += 1
            m = parity1[0], parity1[1], i
            loc.append(m)

        if parity2[0] != parity2[1]:
            change[1] += 1
            n = parity2[0], parity2[1], j
            loc2.append(n)

        parity1[1] = T_FC
        parity2[1] = T_FR

    if (change[0] or change[1]) > 2:
        num = int(max(change[0], change[1]) / 2)
        raise ValueError('Image defect : your image contains at least {:d} blobs!!'.format(num))

    print("Image has been scanned")

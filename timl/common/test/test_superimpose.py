import numpy as np

from timl.segmentation.preprocessing import scan, superimpose
from .test_preprocessing import tofix_test_mask


def tofix_test_overlap(img1,img2):
    tofix_test_mask(im2)
    scan(im2)

    img_new = superimpose(img1, img2)
    arr1 = np.array(img_new)
    arr2 = np.array(img2)

    t = list()

    # here we test for matching black mask
    # using just 1 channel for the rgb image, making comparison holding
    # each row index constant and checking across its column values
    for i in range(arr2.shape[0]):
        e1 = list(np.where(arr1[i, :, 0] == 0)[0])
        e2 = list(np.where(arr2[i, :] == 0)[0])
        t.append(e2 == np.intersect1d(e1, e2).tolist())

    assert np.all(t), "The index of black background for the masks should s" \
                  "correspond with the superimposed image "


if __name__ == "__main__":

    fp = input("Enter image path: ")
    fp2 = input("Enter its mask path: ")

    im = tar2.get_image(fp)
    im2 = tar2.get_image(fp2)

    test_overlap(im, im2)
    print()
    print("Completed !!!")


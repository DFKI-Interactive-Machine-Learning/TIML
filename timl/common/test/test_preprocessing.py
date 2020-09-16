from PIL import Image
import numpy as np

from .preprocessing import scan, extension


def get_image(fp: str):
    img = Image.open(fp)

    return img


def get_value(img: Image) -> int:
    a_im = np.array(img)
    white_pixel = np.where(a_im == 255)

    cnt = 0
    for i in white_pixel:
        cnt += len(i)

    return cnt


def tofix_test_mask(img: Image):
    assert np.unique(img).tolist() == [0, 255], "mask should contain values 0 and 255 only"


def tofix_test_extension(im: Image, fct: float):
    assert fct > 1, "Extension factor should be greater than 1"

    im_ex = extension(im, fct)
    num1 = get_value(im_ex)
    num2 = get_value(im)

    assert num1 > num2, "extended mask should have more white pixels"


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    factor = float(input("Enter the extension factor: "))

    im = get_image(image_path)
    test_mask(im)
    scan(im)
    test_extension(im, factor)
    print()
    print("Tests Complete !!!")

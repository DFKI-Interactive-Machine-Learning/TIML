from timl.segmentation.preprocessing import extension
from PIL.Image import Image

from typing import List
from typing import Iterable
from typing import Tuple


def format_array(a, format="{:.3f}"):
    fa = [format.format(v) for v in a]
    return '[' + ','.join(fa) + ']'


def deep_getsizeof(o, ids=None):
    """Find the memory footprint of a Python object
    This is a recursive function that rills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    Adapted from: https://github.com/the-gigi/deep/blob/master/deeper.py#L80

    :param o: the object
    :param ids: The list of ids already visited. Leave it to None for top-level invocation.
    :return: The number of bytes used.
    """

    from collections import Mapping, Iterable
    from sys import getsizeof

    if ids is None:
        ids = set()

    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str):  # or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())

    if isinstance(o, Iterable):
        return r + sum(d(x, ids) for x in o)

    return r


def jaccard_index(v1: Iterable[float], v2: Iterable[float]) -> float:
    """Compute the similarity between two vectors.
    The weighted Jaccard index of similarity between two vectors is defined as:
    Jsimilarity = sum(min(b1_i, b2_i)) / sum(max(b1_i, b2_i)
    where v1_i, and v2_i are the ith element of the two input vectors.

    The two vectors must have the same length.

    See: https://en.wikipedia.org/wiki/Jaccard_index
    Where it is also called Ruzicka similarity."""

    #if len(v1) != len(v2):
    #    raise Exception("The two input vectors must have the same lenght. Found {} and {}".format(len(v1), len(v2)))

    sum_mins = 0
    sum_maxs = 0

    for f1, f2 in zip(v1, v2):
        min_b = min(f1, f2)
        sum_mins += min_b
        max_b = max(f1, f2)
        sum_maxs += max_b

    j = sum_mins / sum_maxs

    return j


def jaccard_similarity(im1: Image, im2: Image) -> float:
    """Compute the similarity between two images.
    First, for each image an histogram of the pixels distribution is extracted.
    Then, the similarity between the histograms is compared using the weighted Jaccard index of similarity, defined as:
    Jsimilarity = sum(min(b1_i, b2_i)) / sum(max(b1_i, b2_i)
    where b1_i, and b2_i are the ith histogram bin of images 1 and 2, respectively.

    The two images must have same resolution and number of channels (depth).

    See: https://en.wikipedia.org/wiki/Jaccard_index
    Where it is also called Ruzicka similarity."""

    if im1.size != im2.size:
        raise Exception("Images must have the same size. Found {} and {}".format(im1.size, im2.size))

    n_channels_1 = len(im1.getbands())
    n_channels_2 = len(im2.getbands())
    if n_channels_1 != n_channels_2:
        raise Exception("Images must have the same number of channels. Found {} and {}".format(n_channels_1, n_channels_2))

    assert n_channels_1 == n_channels_2

    sum_mins = 0
    sum_maxs = 0

    hi1 = im1.histogram()  # type: List[int]
    hi2 = im2.histogram()  # type: List[int]

    # Since the two images have the same amount of channels, they must have the same amount of bins in the histogram.
    assert len(hi1) == len(hi2)

    for b1, b2 in zip(hi1, hi2):
        min_b = min(b1, b2)
        sum_mins += min_b
        max_b = max(b1, b2)
        sum_maxs += max_b

    jaccard_index = sum_mins / sum_maxs

    return jaccard_index


def generate_circular_mask_bresenham(size: Tuple[int, int]) -> Image:
    """Given a resolution, generate a 'mask' image with a circle.
    Image mode will be mono-channel 'L'.
    Internal in-circle pixels will be white (255), while external pixels will be black (0)."""

    import PIL.Image
    import numpy as np

    w = size[0]
    h = size[1]

    a = np.zeros(shape=(w, h), dtype=np.uint8)

    center_x = int(w / 2)
    center_y = int(h / 2)

    radius = min(center_x, center_y)

    min_dim = min(w, h)
    if min_dim % 2 == 0:  # if it is an even number
        radius -= 1  # remove 1 pixel in order to avoid out of bounds

    radius_sq = radius * radius


    """
    From: https://stackoverflow.com/questions/1201200/fast-algorithm-for-drawing-filled-circles
    int largestX = circle.radius;
    for (int y = 0; y <= radius; ++y) {
        for (int x = largestX; x >= 0; --x) {
            if ((x * x) + (y * y) <= (circle.radius * circle.radius)) {
                drawLine(circle.center.x - x, circle.center.x + x, circle.center.y + y);
                drawLine(circle.center.x - x, circle.center.x + x, circle.center.y - y);
                largestX = x;
                break; // go to next y coordinate
            }
        }
    }
    """

    largest_x = radius
    for y in range(radius+1):
        for x in range(largest_x, -1, -1):
            if (x * x) + (y * y) <= radius_sq:

                # These are to draw the circle border
                # a[center_x - x, center_y - y] = 255
                # a[center_x + x, center_y - y] = 255
                # a[center_x - x, center_y + y] = 255
                # a[center_x + x, center_y + y] = 255

                # There are to draw a filled circle
                # drawLine(circle.center.x - x, circle.center.x + x, circle.center.y + y);
                for col in range(center_x - x, center_x + x + 1):
                    a[col, center_y + y] = 255
                # drawLine(circle.center.x - x, circle.center.x + x, circle.center.y - y);
                for col in range(center_x - x, center_x + x + 1):
                    a[col, center_y - y] = 255

                largest_x = x
                break  # go to next y coordinate

    out = PIL.Image.fromarray(obj=a.T, mode='L')
    out = extension(out, 1.05)

    return out


def generate_circular_mask(size: Tuple[int, int]) -> Image:
    """Given a resolution, generate a 'mask' image with a circle.
    Image mode will be mono-channel 'L'.
    Internal in-circle pixels will be white (255), while external pixels will be black (0)."""

    import PIL.Image
    import numpy as np

    w = size[0]
    h = size[1]

    a = np.zeros(shape=(w, h), dtype=np.uint8)

    center_x = w / 2.0
    center_y = h / 2.0

    radius = min(center_x, center_y)
    radius_sq = radius * radius

    """
    From: https://stackoverflow.com/questions/1201200/fast-algorithm-for-drawing-filled-circles
    for(int y=-radius; y<=radius; y++)
        for(int x=-radius; x<=radius; x++)
            if(x*x+y*y <= radius*radius)
                setpixel(origin.x+x, origin.y+y);
    """

    for y in range(0, h):
        dy = y - center_y
        dy_sq = dy * dy
        for x in range(0, w):
            dx = x - center_x
            if (dx * dx) + dy_sq <= radius_sq:
                a[x, y] = 255

    out = PIL.Image.fromarray(obj=a.T, mode='L')

    return out


def generate_rectangular_mask(size: Tuple[int, int], factor=0.7) -> Image:
    import numpy as np
    from PIL import Image

    r = size[0]
    c = size[1]
    # reverse dimensions for numpy array and image
    plc_holder = np.zeros(shape=(c, r), dtype=np.uint8)

    h, w = plc_holder.shape

    w_center = w // 2
    h_center = h // 2

    w_ratio = w * factor
    h_ratio = h * factor

    w_low_index = w_ratio // 2
    w_high_index = np.ceil(w_ratio / 2.0)

    h_low_index = h_ratio // 2
    h_high_index = np.ceil(h_ratio / 2.0)

    row_low = int(h_center - h_low_index)
    row_high = int(h_center + h_high_index)

    col_low = int(w_center - w_low_index)
    col_high = int(w_center + w_high_index)

    plc_holder[row_low:row_high, col_low:col_high] = 255.0

    out = Image.fromarray(obj=plc_holder, mode='L')

    return out


def generate_random_image(size: Tuple[int, int]) -> Image:
    """Generate a uniform-noise RGB image, with all pixels in full range 0-255"""

    import PIL.Image

    from numpy.random import rand

    w = size[0]
    h = size[1]

    # Generate random with range [0.0,256.0)
    a_float = rand(h, w, 3) * 256

    a_int = a_float.astype(dtype='uint8')

    out = PIL.Image.fromarray(obj=a_int, mode='RGB')

    return out

# Classes to load and augment the images listed in a training dataframe.
# We use the Decorator design pattern:
# https://en.wikipedia.org/wiki/Decorator_pattern

import PIL
from PIL.Image import Image

from abc import abstractmethod
from abc import ABC
from collections import namedtuple, OrderedDict
from typing import List
from typing import Tuple
from typing import Optional

IMAGE_AUGMENTATION_PRESETS = ["none", "hflip", "hflip_rot24", "hflip_rot4"]


class ImageProvider(ABC):
    """This is the top-level interface of the Decorator pattern."""

    @abstractmethod
    def num_images(self) -> int:
        pass

    @abstractmethod
    def get_image(self, i: int) -> Image:
        pass


class ImageAugmenter(ImageProvider):
    """This is the abstract decorator class of the Decorator design pattern."""

    def __init__(self, provider: ImageProvider):
        self.augmenter = provider

    @abstractmethod
    def num_images(self) -> int:
        pass

    @abstractmethod
    def get_image(self, i: int) -> Image:
        pass


class SingleImageProvider(ImageProvider):
    """Dummy augmenter of images. Returns the same image given as input."""

    def __init__(self, image: Image):
        self.image = image

    def num_images(self) -> int:
        return 1

    def get_image(self, i: int) -> Image:
        if i != 0:
            raise Exception("Image index must be 0 for this class")

        return self.image


class MultipleImageProvider(ImageProvider):

    def __init__(self, image_paths: List[str], size: Optional[Tuple[int, int]], resample_filter: Optional[int], color_space: Optional[str]):
        """

        :param image_paths:
        :param size: Optionally, them image can be resized just after loading.
        :param resample_filter: a PIL filter. See https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#concept-filters
        :param color_space: Optionally, an image color space/mode can be changed just after resizing. See: <https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#concept-modes>
        """
        self._image_paths = image_paths
        self._size = size
        self._filter = resample_filter
        self._color_space = color_space

    def num_images(self) -> int:
        return len(self._image_paths)

    def get_image(self, i: int) -> Image:
        img = PIL.Image.open(self._image_paths[i])  # type: Image
        if self._size is not None:
            img = img.resize(self._size, resample=self._filter)

        # We support grey images, but normalize them to RGB
        if img.mode == 'L':
            img = img.convert(mode='RGB')

        if img.mode != 'RGB':
            raise Exception("Expecting to load images in RGB format. Found {}".format(img.mode))

        if self._color_space != 'RGB':
            # print("MultipleImageProvider: Converting to mode " + str(self._color_space))
            img = img.convert(mode=self._color_space)

        return img


class HFlipImageAugmenter(ImageAugmenter):
    """Decorator doubling the images by performing an horizontal flip."""

    def __init__(self, provider: ImageProvider):
        super().__init__(provider=provider)

    def num_images(self) -> int:
        return self.augmenter.num_images() * 2

    def get_image(self, i: int) -> Image:
        if i % 2 == 0:
            # print("NoFlip")
            return self.augmenter.get_image(i // 2)
        elif i % 2 == 1:
            # print("Flip")
            original_img = self.augmenter.get_image(i // 2)
            flipped_img = original_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            return flipped_img


class RotatedImageAugmenter(ImageAugmenter):
    """Decorator returning the image rotated of many steps."""

    def __init__(self, provider: ImageProvider, rot_steps: int):
        super().__init__(provider=provider)
        self.steps = rot_steps

    def num_images(self) -> int:
        return self.augmenter.num_images() * self.steps

    def get_image(self, i: int) -> Image:

        orig_img = self.augmenter.get_image(i // self.steps)
        rot_step = i % self.steps
        if rot_step == 0:
            return orig_img
        else:
            angle = 360 * rot_step / self.steps
            # print("Rotating {}".format(angle))
            return orig_img.rotate(angle=angle, expand=False)


CacheStats = namedtuple("CacheStats", ["requests", "hits"])


class CachingImageAugmenter(ImageAugmenter):
    """Decorator which is retaining in memory cache already requested images. Prevents reloading/reprocessing.
    The images are store using an LRU (Least Recently Used) strategy.
    LRU is implemented using an OrderedDict.
    See: https://docs.python.org/3/library/collections.html#collections.OrderedDict"""

    def __init__(self, provider: ImageProvider, cache_limit: int):
        super().__init__(provider=provider)
        self.cache_limit = cache_limit

        self.cache = OrderedDict()

        self.requests = 0
        self.hits = 0

    def __del__(self):
        print("Cache stats on deletion: objid={}, requests={}, hits={}%".format(id(self), self.requests, self.hits))

    def num_images(self) -> int:
        return self.augmenter.num_images()

    def get_image(self, i: int) -> Image:
        # now = time.time()
        self.requests += 1

        if i in self.cache:
            self.hits += 1
            out = self.cache[i]
            # move the image entry to the end, as it is the most recently used
            self.cache.move_to_end(i)
        else:
            # Remove old images if the cache is too big
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)

            out = self.augmenter.get_image(i)
            self.cache[i] = out

        #import threading
        #thr_id = threading.current_thread().getName()
        #myid = id(self)
        #print("Cache stats: thread={}, objid={}, asked={}, requests={}, hits={}%".format(thr_id, myid, i, self.requests, 100.0 * self.hits / self.requests))

        return out

    def get_stats(self) -> CacheStats:
        return CacheStats(requests=self.requests, hits=self.hits)


#
# Converts filter options to integers
FILTER_MAP = {
    "nearest": PIL.Image.NEAREST,
    "bilinear": PIL.Image.BILINEAR,
    "bicubic": PIL.Image.BICUBIC,
    "lanczos": PIL.Image.LANCZOS
}


def image_provider_factory(config: str,
                           image_paths: List[str],
                           resize: Optional[Tuple[int, int]],
                           resize_filter: Optional[str],
                           color_space: Optional[str],
                           image_cache_limit: int = 0) -> ImageProvider:
    """
    THis is the factory method to compose a chain of image augmenters.

    :param config: The name of a preset configuration: "none", "hflip", "hflip_rot24"
    :param image_paths: A list of file paths. Each path refers to an image to load and process.
    :param resize: If not None, loaded images will be resized to the specified resolution.
    :param resize_filter: When resizing, here you decide which resize filter to use: nearest, bilinear, bicubic, lanczos.
    :param image_cache_limit: The maximum number of images to cache, to avoi reloading from disk.
     Use 0 to disable caching completely.
    :param color_space: If not None, the loaded image will be converted into another color space.
     See Pillow documentation on color modes: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    :return: The decorated chain of image providers.
    """

    if resize is None:
        resize_filter_pil = None
    else:
        if resize_filter not in FILTER_MAP:
            raise Exception("Invalid resizefilter {}. Please, choose among {}".format(resize_filter, FILTER_MAP.keys()))
        resize_filter_pil = FILTER_MAP[resize_filter]

    provider = MultipleImageProvider(image_paths=image_paths,
                                     size=resize, resample_filter=resize_filter_pil,
                                     color_space=color_space)

    # If caching is requested, decorate the basic provider with a Caching augmenter
    if image_cache_limit != 0:
        print("Image Cache: Using size {}".format(image_cache_limit))
        provider = CachingImageAugmenter(provider=provider, cache_limit=image_cache_limit)
    else:
        print("Image Cache: Disabled!")

    if config == "none":
        out = provider
    elif config == "hflip":
        out = HFlipImageAugmenter(provider=provider)
    elif config == "hflip_rot4":
        out = RotatedImageAugmenter(rot_steps=4,
                                    provider=HFlipImageAugmenter(provider=provider))
    elif config == "hflip_rot24":
        out = RotatedImageAugmenter(rot_steps=24,
                                    provider=HFlipImageAugmenter(provider=provider))

    else:
        assert config not in IMAGE_AUGMENTATION_PRESETS
        raise Exception("Augmentation preset {} not recognized.".format(config))

    return out



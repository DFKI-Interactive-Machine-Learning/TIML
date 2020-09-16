import numpy as np
import keras

from timl.common.imageaugmentation import image_provider_factory
import os
import math
import pandas

from typing import Optional
from typing import Tuple
from typing import List
from typing import Dict

IMAGES_COLUMN = 'image_name'
IMAGES_COLUMN_META = "image_name"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


class SkincareDataGenerator(keras.utils.Sequence):
    """This class implements the Keras data generator.
    It is the omni-comprehensive data generator able to:
    - provide images for testing
    - provide images together with ground truth for training
    - use cached activation values instead of images
    - append additional metadata to the images (or the activation values)

    In addition to implementing the methods needed by the generator, this class
    performs several checks about the existence and consistency of all the expected data. Among them:
    - It will check that all the image files are present in the disk.
    - It used, that all the activation files are present on disk
    - That the 1-hot representation is valid (lines sums to 1, Only unit value)
      - The metadata do not have to necessarily be in 1-hot format.

    In its default configuration (passing only mandatory parameters), the generator is good for testing.
    Hence, the output of the __get_item__() is a single element.
    When specifying ground truth and augmentation, it is good for training, the items are a pair (x, y),
    where x is the numpy format of the normalized images, and y the ground truth.
    When specifying activation, x will be a cached activation vector.
    When specifying metadata, returned items will be ([x, m], y)
    Where m is the vector of metadata.
    """

    def __init__(self,
                 images_df: pandas.DataFrame,
                 images_dir: str,

                 image_size: Tuple[int, int],
                 resize_filter: str,
                 color_space: str,
                 batch_size: int,

                 image_augmentation: str = "none",

                 activations_dir: Optional[str] = None,
                 activations_size: Optional[int] = None,

                 truth_df: Optional[pandas.DataFrame] = None,
                 truth_columns: Optional[List[str]] = None,

                 meta_df: Optional[pandas.DataFrame] = None,
                 meta_columns: Optional[List[str]] = None,

                 shuffle: bool = False,
                 is_multilabel: bool = False
                 ):  # Added a new parameter classes

        """

        :param images_df: The dataframe containing the 'image_name' column, listing all the images to be used
        :param images_dir: The directory where to look for the images. Image name must be the same as column 'image_name', followed by a common extension.
        :param image_size: A 2-tuple like (640,480). Images will be scaled to this resolution after loading
        :param resize_filter: A filter as supported by PIL: 'nearest', 'bilinear', 'bicubic', 'lanczos'
        :param color_space: Images will be converted into a PIL format like 'RGB', 'HSV', ...
        :param batch_size: The size of the batch returned by __get_item__()
        :param image_augmentation: Whether to augment the set of images. See presets in timl.common.imageaugmentation.image_provider_factory()
        :param activations_dir: The directory where to look for activation cache. If not None, this cache will be used instead
        of the real image. There ust be a file with image_name-aug.npy for each image mentioned in the images_df.
        aug is a 10 digit code with the results of the augmented versions of the given image.
        :param activations_size: The number of elements in each activations file.
        :param truth_df: The dataframe with 'image_name' (will be used as index) and the columns with ground truth training data.
        :param truth_columns: The list of columns to be used as ground truth.
        :param meta_df: The dataframe with 'image_name' (will be used as index) and the columns with metadata.
        :param meta_columns: The list of columns to be used from teh metadata dataframe.
        :param shuffle: When True, the data passed through the generator will be shuffled. Likely needed for training.
        """

        self._images_df = images_df
        self._image_dir = images_dir

        self._image_size = image_size
        self._batch_size = batch_size
        self._n_channels = 3  # TODO -- this should be taken from the image dataset
        self._shuffle = shuffle

        #
        # Info for generation
        #

        if IMAGES_COLUMN not in self._images_df.columns:
            raise ValueError("Missing column '{}' in images dataframe.".format(IMAGES_COLUMN))

        # Number of images in this dataset
        self._n_images = self._images_df.shape[0]

        #
        # Scan for the existence of all images
        #

        # A list of paths (name + extension) to the images
        self._image_paths = []  # type: List[str]

        #
        # Scan for the existence of the images
        #
        for img_name in self._images_df.image_name:
            ext_found = False
            for ext in IMAGE_EXTENSIONS:
                full_path = os.path.join(self._image_dir, img_name + ext)
                if os.path.exists(full_path):
                    self._image_paths.append(full_path)
                    ext_found = True
                    break  # Jumps out of the extension loop and go to the next image

            if not ext_found:
                raise ValueError("No image file found for " + img_name)

        # Instantiate the ImageGenerator
        self._img_provider = image_provider_factory(config=image_augmentation,
                                                    image_paths=self._image_paths,
                                                    resize=image_size,
                                                    resize_filter=resize_filter,
                                                    color_space=color_space)

        # Number of "simulated" images
        self._n_images_augmented = self._img_provider.num_images()  # self.df.shape[0]

        # augmented / real images ratio
        self._augmentation_ratio = int(self._n_images_augmented / self._n_images)

        #
        # Activation
        #
        if activations_dir is None:
            self._activations_dir = None
        else:

            self._activations_dir = activations_dir

            if activations_size is None:
                raise ValueError("Activation size must be specified. Found None.")
            self._activations_size = activations_size

            #
            # Check for the existence of all data

            # Scan directory and retrieve only files ending with ".npy"
            all_dir_files = os.listdir(self._activations_dir)
            # print("Found {} files in activation dir".format(len(all_dir_files)))
            self._activation_files = []
            for fname in all_dir_files:
                if fname.endswith(".npy"):
                    self._activation_files.append(fname)
            print("Found {} activation files".format(len(self._activation_files)))

            #
            # Compose the lookup dictionary mapping an image_name to the list of (augmented) activations
            self._activation_map = {}  # type: Dict[str, List[str]]

            image_name_set = {im for im in self._images_df[IMAGES_COLUMN]}
            for fname in self._activation_files:

                # Split the filename to gather the augmentation id
                image_name, augid_and_ext = fname.split("-")
                assert len(augid_and_ext) == 10 + 4  # the code plus ".npy"
                assert augid_and_ext.endswith(".npy")

                if image_name not in image_name_set:  # self._df["image_name"]:
                    continue

                aug_id_str, _ = augid_and_ext.split(".")
                assert len(aug_id_str) == 10
                aug_id = int(aug_id_str)

                if aug_id >= self._augmentation_ratio:
                    continue

                # Ensure we have an entry
                # For each image we build a list ready to accommodate the filename for its augmented version
                if image_name not in self._activation_map:
                    self._activation_map[image_name] = ["n/a"] * self._augmentation_ratio

                # Fill the map
                self._activation_map[image_name][aug_id] = fname

            # Consistency check: all images mentioned in the dataframe must have all entries filled
            na_entries = []
            for img in self._images_df[IMAGES_COLUMN]:

                # If the image is not in the map at all, put a dummy aug_idx and continue
                if img not in self._activation_map:
                    na_entries.append((img, -1))
                    continue

                img_entry = self._activation_map[img]
                assert len(img_entry) == self._augmentation_ratio

                # All of the augmentation files must be there
                for i, activation_file in enumerate(img_entry):
                    if activation_file == "n/a":
                        na_entries.append((img, i))

            if len(na_entries) > 0:
                print("Missing images: ")
                print(na_entries)
                raise Exception(
                    "Missing activation files for {} images. See list in console.".format(len(na_entries)))

            assert len(self._activation_map) == self._n_images

        #
        # Info for training
        #

        # By default, the classes is None, so no ground truth info will be provided.
        self._all_classes = None  # type: Optional[List[str]]

        if truth_df is None:
            self._truth_df = None

        else:
            if IMAGES_COLUMN not in truth_df.columns:
                raise Exception("Missing column {} in ground truth dataframe.".format(IMAGES_COLUMN))

            self._truth_df = truth_df.set_index(IMAGES_COLUMN)

            # Extract information for training (ground truth)
            self._all_classes = truth_columns

            # Count the number of classes from classes string
            self._n_classes = len(self._all_classes)

            for cls in self._all_classes:
                if cls not in self._images_df.columns:
                    raise ValueError("Column {} is not in the ground truth dataframe".format(cls))

            #
            # One-hot consistency check
            #

            # Count the number of occurrences of ones and zeroes
            cls_frame = self._truth_df[self._all_classes]  # extract only columns for the classes
            n_zeros = (cls_frame == 0).astype(int).sum(axis=1).to_numpy()
            n_ones = (cls_frame == 1).astype(int).sum(axis=1).to_numpy()

            if is_multilabel:

                # Verify that the sum of ones and zeroes is equal to the number of classes
                #sum_ones = cls_frame.sum(axis=1).to_numpy()  # sum by line
                #if not np.array_equal(sum_onehot, ones):
                #    raise ValueError("Some of the rows do not sum to 1.")

                n_tot = n_zeros + n_ones
                # print(n_tot)
                n_classes_per_row = [self._n_classes] * len(cls_frame)
                # print(ref)
                if not np.array_equal(n_tot, n_classes_per_row):
                    raise ValueError("Problem with label encoding."
                                     "In at least 1 line, the sum of 1s and 0s is not like the number of classes."
                                     "Possibly, some values are neither 0 nor 1 ")

            else:

                ones = np.ones(cls_frame.shape[0])  # a column of ones.

                # Verify that each one-hot row sums to 1
                #sum_onehot = cls_frame.sum(axis=1).to_numpy()  # sum by line
                #if not np.array_equal(sum_onehot, ones):
                #    raise ValueError("Some of the rows do not sum to 1.")

                if not np.array_equal(n_zeros, ones * (self._n_classes - 1)):
                    raise ValueError("Problem with the one-hot vector in data-frame: All lines must have {} zeros.".format(
                        self._n_classes - 1))

                if not np.array_equal(n_ones, ones):
                    raise ValueError("Problem with the one-hot vector in data-frame: All lines must have exactly one 1.")

        #
        # METADATA
        #
        if meta_df is None:
            self._metadata_df = None

        else:
            if IMAGES_COLUMN_META not in meta_df.columns:
                raise Exception("Missing column {} in metadata dataframe.".format(IMAGES_COLUMN_META))

            self._metadata_df = meta_df.set_index(IMAGES_COLUMN_META)
            self._metadata_columns = meta_columns

            for md in self._metadata_columns:
                if md not in self._metadata_df.columns:
                    raise ValueError("Column {} is not in the metadata dataframe".format(md))

        #
        # Batch management
        #

        # Denotes the number of batches per epoch
        self._n_batches = int(math.ceil(self._n_images_augmented / self._batch_size))

        # The sequence of indices (potentially shuffled) that will be used in the batch generation
        self._indices = None  # type: Optional[np.ndarray]

        # First initialization of the indices
        self.on_epoch_end()

    def __len__(self):
        return self._n_batches

    def __getitem__(self, batch_index):
        """Generate one batch of data"""

        # Retrieve indexes of the batch
        image_indices = self._indices[batch_index * int(self._batch_size): (batch_index + 1) * int(self._batch_size)]

        #  Don't worry about index overflow. The splitter will stop to the array limit.
        # image_indices is eventually shorter than batch_size
        assert len(image_indices) <= self._batch_size

        #
        # IMAGES
        if self._activations_dir is not None:
            out = self._get_activations_batch(activation_indices=image_indices)
        else:
            out = self._get_img_batch(image_indices)

        # Compute the original indices, and image names, before augmentation
        original_indices = [int(idx / self._augmentation_ratio) for idx in image_indices]
        original_image_names = self._images_df.iloc[original_indices][IMAGES_COLUMN]

        #
        # META DATA
        if self._metadata_df is not None:
            m_df = self._metadata_df.loc[original_image_names][self._metadata_columns]
            m = m_df.to_numpy()
            # Concatenate image data with metadata
            out = [out, m]

        #
        # Ground truth
        if self._truth_df is not None:
            y = self._truth_df.loc[original_image_names][self._all_classes].to_numpy()
            # Append ground truth data to the output
            out = out, y

        return out

    def get_dataframe(self) -> pandas.DataFrame:
        return self._images_df

    def get_ground_truth_dataframe(self) -> pandas.DataFrame:
        """Return the dataframe containing only the columns used for training."""

        # Resort according to the main dataframe
        images = self._images_df[IMAGES_COLUMN]

        out_frame = self._truth_df.loc[images][self._all_classes]  # extract only columns for the classes

        assert len(out_frame) == self._n_images
        assert out_frame.shape[1] == len(self._all_classes)

        return out_frame

    def get_image_names(self) -> List[str]:
        return self._images_df[IMAGES_COLUMN].to_list()

    def get_n_images(self) -> int:
        return self._n_images

    def get_n_images_augmented(self) -> int:
        return self._n_images_augmented

    def get_image_size(self) -> Tuple[int, int]:
        return self._image_size

    def get_augmentation_ratio(self) -> int:
        return self._augmentation_ratio

    def get_n_batches(self) -> int:
        return self._n_batches

    def get_class_proportion(self) -> List[float]:

        freq_count = self._truth_df[self._all_classes].sum(axis=0)
        freq_count = freq_count / self._n_images
        return freq_count.to_numpy().tolist()

    def get_image_paths(self):
        return self._image_paths

    def get_class_names(self) -> List[str]:
        return self._all_classes

    def get_class_count(self) -> int:
        return self._n_classes

    def on_epoch_end(self) -> None:
        """Updates indices after each epoch"""

        # By default creates a list of sequential indices
        self._indices = np.arange(self._n_images_augmented)
        # If asked, shuffle them
        if self._shuffle:
            np.random.shuffle(self._indices)

    def _get_img_batch(self, image_indices) -> np.ndarray:

        """
        :param image_indices: the list of integer indices to use on the images dataset.
        :return: a numpy ndarray of shape (m, w, h, 3), where m is len(image_indices), wxh is the image size, 3 is for RGB channels.
        """

        # Prepare a zero-filled array that will store all the images
        x_data = np.zeros((len(image_indices), *self._image_size, self._n_channels))

        # get each image and put it in the output array
        for i, idx in enumerate(image_indices):
            im_ = self._img_provider.get_image(idx)
            x_data[i, ] = np.array(im_, dtype=np.float32)

        # Normalize everything
        x_data /= 255.0

        return x_data

    def _get_activations_batch(self, activation_indices):

        # Prepare the array of activation values
        x_data = np.zeros((len(activation_indices), self._activations_size))

        # For each requested index, retrieve the activations
        for i, idx in enumerate(activation_indices):

            # Compute original index and reminder (augmentation index)
            original_idx = int(idx / self._augmentation_ratio)
            aug_idx = idx % self._augmentation_ratio
            assert 0 <= aug_idx <= self._augmentation_ratio

            img_name = self._images_df.iloc[original_idx][IMAGES_COLUMN]

            activation_file = self._activation_map[img_name][aug_idx]
            activation_path = os.path.join(self._activations_dir, activation_file)
            activations = np.load(activation_path)

            if len(activations) != self._activations_size:
                raise ValueError("Size of the feature vector expected to be {}. In file '{}' found {}.".format(self._feature_size, activation_path, len(activations)))

            x_data[i, ] = np.array(activations)

        return x_data

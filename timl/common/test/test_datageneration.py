import pytest

import os
import pandas

from ..datageneration import SkincareDataGenerator

import pkg_resources

#
# A couple of small datasets for testing
ISIC2019_TRAIN_DF = pkg_resources.resource_filename("timl.data", "ISIC2019/ISIC_2019_Training_GroundTruth_meta-train.csv")
#ISIC2019_DEV_DF = pkg_resources.resource_filename("timl.data", "ISIC2019/ISIC_2019_Training_GroundTruth_meta-dev.csv")
#ISIC2019_TEST_DF = pkg_resources.resource_filename("timl.data", "ISIC2019/ISIC_2019_Training_GroundTruth_meta-test.csv")
ISIC2019_COLUMNS = "MEL,NV,BCC,AK,BKL,DF,VASC,SCC".split(',')

#
# The dataset with the metadata
ISIC2019_META_DF = pkg_resources.resource_filename("timl.data", "ISIC2019/ISIC_2019_Training_Metadata_1hot.csv")
ISIC2019_META_COLUMNS = "age,anterior torso,head/neck,lateral torso,lower extremity,oral/genital,palms/soles,posterior torso,upper extremity,female,male".split(',')

# Directory with jpeg images
ISIC2019_IMAGES_DIR = pkg_resources.resource_filename("timl.data", "ISIC2019/images")

# Directory with the binary numpy vectors extracted from the 2048-node fully-connected layer of a previous model
ISIC2019_ACTIVATIONS_DIR = pkg_resources.resource_filename("timl.data", "ISIC2019/activations")


@pytest.fixture
def trainset() -> pandas.DataFrame:
    assert os.path.exists(ISIC2019_TRAIN_DF)

    df = pandas.read_csv(ISIC2019_TRAIN_DF)

    # Original columns
    assert "image_name" in df
    assert len(df.columns) > 1

    for col in ISIC2019_COLUMNS:
        assert col in df

    df = df.head(n=10)

    return df


@pytest.fixture
def metadataset() -> pandas.DataFrame:
    assert os.path.exists(ISIC2019_META_DF)

    df = pandas.read_csv(ISIC2019_META_DF)

    # Original columns
    assert "image_name" in df
    assert len(df.columns) > 1

    for col in ISIC2019_META_COLUMNS:
        assert col in df

    return df


def test_skincare_datagen_prediction(trainset):
    images_dir = ISIC2019_IMAGES_DIR

    augmentation = 'hflip_rot24'
    batch_size = 32
    img_size = (227, 227)

    skgen = SkincareDataGenerator(images_df=trainset,
                                  images_dir=images_dir,
                                  image_size=img_size,
                                  resize_filter="nearest",
                                  color_space='RGB',
                                  batch_size=batch_size,
                                  image_augmentation=augmentation
                                  )

    n_images = skgen.get_n_images()
    n_aug_images = skgen.get_n_images_augmented()
    n_batches = skgen.get_n_batches()
    assert n_batches <= n_aug_images
    assert len(skgen) == n_batches
    assert skgen.get_augmentation_ratio() >= 1

    assert (augmentation == 'hflip_rot24' and n_aug_images == n_images * 48)

    for b in range(n_batches):
        batch = skgen.__getitem__(b)
        assert batch is not None
        assert len(batch.shape) == 4
        assert batch.shape[0] <= batch_size
        assert batch.shape[1] == img_size[0]
        assert batch.shape[2] == img_size[1]
        assert batch.shape[3] == 3


def test_skincare_datagen_prediction_activations(trainset):

    augmentation = 'hflip'
    batch_size = 32
    img_size = (227, 227)
    activation_size = 2048

    skgen = SkincareDataGenerator(images_df=trainset,
                                  images_dir=ISIC2019_IMAGES_DIR,
                                  image_size=img_size,
                                  resize_filter="nearest",
                                  color_space='RGB',
                                  batch_size=batch_size,
                                  image_augmentation=augmentation,
                                  activations_dir=ISIC2019_ACTIVATIONS_DIR,
                                  activations_size=activation_size
                                  )

    n_images = skgen.get_n_images()
    n_aug_images = skgen.get_n_images_augmented()
    n_batches = skgen.get_n_batches()
    assert n_batches <= n_aug_images
    assert len(skgen) == n_batches
    assert skgen.get_augmentation_ratio() >= 1

    assert (augmentation == 'hflip' and n_aug_images == n_images * 2)

    for b in range(n_batches):
        batch = skgen.__getitem__(b)
        assert batch is not None
        assert len(batch.shape) == 2
        assert batch.shape[0] <= batch_size
        assert batch.shape[1] == activation_size


def test_skincare_datagen_train_with_metadata(trainset, metadataset):
    truth_df = trainset

    augmentation = 'hflip_rot24'
    batch_size = 32
    img_size = (227, 227)

    skgen = SkincareDataGenerator(images_df=trainset,
                                  images_dir=ISIC2019_IMAGES_DIR,
                                  image_size=img_size,
                                  resize_filter="nearest",
                                  color_space='RGB',
                                  batch_size=batch_size,
                                  image_augmentation=augmentation,
                                  truth_df=truth_df,
                                  truth_columns=ISIC2019_COLUMNS,
                                  meta_df=metadataset,
                                  meta_columns=ISIC2019_META_COLUMNS,
                                  shuffle=True
                                  )

    n_images = skgen.get_n_images()
    n_aug_images = skgen.get_n_images_augmented()
    n_batches = skgen.get_n_batches()
    assert n_batches <= n_aug_images
    assert len(skgen) == n_batches
    assert skgen.get_augmentation_ratio() >= 1

    assert (augmentation == 'hflip_rot24' and n_aug_images == n_images * 48)

    for b in range(n_batches):
        batch = skgen.__getitem__(b)
        assert type(batch) == tuple
        assert len(batch) == 2

        batch_data, batch_truth = batch

        assert len(batch_data) == 2
        assert type(batch_data) == list

        batch_img, batch_md = batch_data

        assert len(batch_img.shape) == 4
        assert batch_img.shape[0] <= batch_size
        assert batch_img.shape[1] == img_size[0]
        assert batch_img.shape[2] == img_size[1]
        assert batch_img.shape[3] == 3

        assert len(batch_md.shape) == 2
        assert batch_md.shape[0] <= batch_size
        assert batch_md.shape[1] <= len(ISIC2019_META_COLUMNS)

        assert len(batch_truth.shape) == 2
        assert batch_truth.shape[0] <= batch_size
        assert batch_truth.shape[1] == len(ISIC2019_COLUMNS)

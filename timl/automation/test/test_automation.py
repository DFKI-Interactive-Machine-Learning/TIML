import os

import pandas

import pytest

from typing import List

SAMPLE_DATAFRAME_FILEPATH = "../data/automation_test_input_1.csv"


@pytest.fixture
def sample_dataframe() -> pandas.DataFrame:

    assert os.path.exists(SAMPLE_DATAFRAME_FILEPATH)
    out = pandas.read_csv(SAMPLE_DATAFRAME_FILEPATH)

    return out


@pytest.fixture
def all_sample_dataframes() -> List[pandas.DataFrame]:
    import glob

    out = []
    dataframe_filenames = glob.glob("../data/automation_test_input_*.csv")
    for fn in dataframe_filenames:
        # print("Adding file " + fn)
        out.append(pandas.read_csv(fn))

    return out


def test_dataframe_input_values(all_sample_dataframes: List[pandas.DataFrame]):
    from timl.common.imageaugmentation import IMAGE_AUGMENTATION_PRESETS

    from ..__main__ import DF_COLUMNS
    from ..__main__ import DF_INPUT_COLUMNS

    for sample_dataframe in all_sample_dataframes:

        #
        # All the columns must be present
        df_column_names = list(sample_dataframe.columns)  # type: List[str]
        for df_col in df_column_names:
            assert df_col in DF_COLUMNS

        #
        # For each row, all input columns must be filled
        for idx in sample_dataframe.index:

            row = sample_dataframe.iloc[idx]

            # All these inputs must be present
            for field in DF_INPUT_COLUMNS:
                val = row[field]
                assert not pandas.isna(val)
                assert not val == ""

        #
        # Split policy
        for split in sample_dataframe["split"]:
            assert split == "pre" or split.startswith("frac=") or split.startswith("n=")

        #
        # Number of epochs
        for epochs in sample_dataframe["epochs"]:
            assert epochs >= 1

        #
        # augmentation method
        for augmentation_method in sample_dataframe["imgaug"]:
            assert augmentation_method in IMAGE_AUGMENTATION_PRESETS

        #
        # Batch size
        for batch_size in sample_dataframe["batchsize"]:
            assert batch_size >= 1

        #
        # Class columns
        for class_columns in sample_dataframe["classcolumns"]:
            assert class_columns == "ben_mal" or len(class_columns.split(";")) >= 2

        #
        # Class weights
        for class_weights in sample_dataframe["classweights"]:
            assert class_weights == "default" or class_weights == "compute"

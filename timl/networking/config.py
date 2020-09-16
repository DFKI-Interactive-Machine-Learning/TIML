import os
import json
import sys

from typing import Optional, Dict


CONFIG_FILENAME = "server_config.json"

# Triplets of (entry_name, description, default_value)
config_template = [
    ("SKINCARE_DATA_DIR", "Directory containing some binary data needed to test the Skincare project code", "path/to/skincaredata"),
    ("ISIC_DATASET_IMG_DIR", "Directory containing the ISIC images (ISIC_0000000.jpeg, ...). Needed only for training.", "path/to/ISIC/Images/"),
    ("IMAGE_CACHE_LIMIT", "The number of images to cache in an ImageProvider. Avoids reloading from disk. Should not be too high in order to avoid excessive memory allocation. Lower this if you see your memory filling too much. Se to 0 to disable image caching.", 0),
    ("SERVER_CLASSIFICATION_MODEL", "Path to the model file (.h5) to load to perform the classification.", "0-keras_model-20190412-142934.h5"),
    ("SERVER_ISIC2019_CLASSIFICATION_MODEL", "Path to the model file (.h5) to load to perform multi-class classification.", "0-keras_model.h5"),
    ("SERVER_SEGMENTATION_WEIGHTS", "Path to the file (.h5) containing the weights for the segmentation model.", "segmentation_model-weigths.h5"),
    ("SERVER_STATIC_PAGES_DIR", "Path to the directory containing the static files.", "html"),
    ("SERVER_STATIC_PAGES_URL", "URL path to access the static pages.", "html")
]


def print_example_config():
    #
    # Automatic generation of config
    template_config = {}
    for key, description, default_val in config_template:
        template_config[key+"_description"] = description
        template_config[key] = default_val

    print(json.dumps(template_config, indent=4))


homedir_path = os.path.expanduser("~")
home_config_filepath = os.path.join(homedir_path, ".timl", CONFIG_FILENAME)

config = None  # type: Optional[Dict]

print("Searching for " + CONFIG_FILENAME)
if os.path.exists(home_config_filepath):
    with open(home_config_filepath, 'r') as fp:
        config = json.load(fp=fp)
elif os.path.exists(CONFIG_FILENAME):
    with open(CONFIG_FILENAME) as fp:
        config = json.load(fp=fp)

if config is None:
    print("Couldn't find file {} neither in {} nor in the working directory {}. Please create one like the following:".format(CONFIG_FILENAME, home_config_filepath, os.getcwd()))
    print_example_config()
    sys.exit(10)

print("Skincare configuration loaded.")

#
# Test keys presence
for key, _, _ in config_template:
    if key not in config:
        print("Couldn't find expected key {} in loaded config. Please, check according to the following template:".format(key))
        print_example_config()
        exit(10)

expected_keys = set([v[0] for v in config_template])
for k in config.keys():
    if k not in expected_keys:
        if not k.endswith("_description"):
            print("Unneeded key '{}' found in config. Consider removing it.".format(k))


def get_image_cache_limit() -> int:
    return int(config["IMAGE_CACHE_LIMIT"])


def get_isic_base_path() -> str:
    return config["ISIC_DATASET_IMG_DIR"]


def get_server_classification_model_path() -> str:
    return config["SERVER_CLASSIFICATION_MODEL"]


def get_server_isic_2019_classification_model_path() -> str:
    return config["SERVER_ISIC2019_CLASSIFICATION_MODEL"]


def get_segmentation_weights_path() -> str:
    return config["SERVER_SEGMENTATION_WEIGHTS"]


def get_server_static_dir() -> str:
    return config["SERVER_STATIC_PAGES_DIR"]


def get_server_static_url() -> str:
    return config["SERVER_STATIC_PAGES_URL"]


def get_skincare_datadir() -> str:
    return config["SKINCARE_DATA_DIR"]

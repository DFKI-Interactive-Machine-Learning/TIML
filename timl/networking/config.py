import os
import json
import sys

from typing import Optional, Dict


CONFIG_FILENAME = "timl_server_config.json"

# Triplets of (entry_name, description, default_value)
config_template = [
    ("CLASSIFICATION_MODEL", "Path to the model file (.h5) for classification.", "0-keras_model-xxx.h5"),
#    ("SERVER_SEGMENTATION_MODEL", "Path to the model file (.h5) for for segmentation.", "segmentation_model.h5"),
    ("REST_API_URL_PREFIX", "The URL prefix to access the REST-API.", "rest"),
    ("STATIC_PAGES_DIR", "Path to the directory containing the static files.", "html"),
    ("STATIC_PAGES_URL_PREFIX", "The URL prefix to access the static pages.", "web")
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

print("Server configuration loaded.")

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


def get_server_classification_model_path() -> str:
    return config["CLASSIFICATION_MODEL"]


# def get_segmentation_model_path() -> str:
#    return config["SEGMENTATION_MODEL"]

def get_server_rest_url_prefix() -> str:
    return config["REST_API_URL_PREFIX"]


def get_server_static_dir() -> str:
    return config["STATIC_PAGES_DIR"]


def get_server_static_url_prefix() -> str:
    return config["STATIC_PAGES_URL_PREFIX"]

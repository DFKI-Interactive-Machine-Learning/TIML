# TIML project

TIML is the Toolkit for Interactive Machine Learning.

It provides a set of command line tools and a web server to facilitate training an usage of Deep Convolutional Neural Networks for image classification and analysis through eXplainable Artificial Intelligence (XAI) techniques.

This package features:

* A script to train CNN-based models for image classification and labeling
* A script to batch generate predictions
* A script to test models against a testset
  * It includes the generation of saliency- and heat- maps.
* A integrate web server and web pages for the interactive exploration and test of trained models

From a development point of view, this package offers:

* An Object-oriented software framework to easily add new CNN architectures without rewriting any of the data preparation, infer, and testing code.

## Online examples

The TIML toolkit has been used to realize the classification code of the [Skincare](https://medicalcps.dfki.de/?page_id=1056) project. Try it here <http://dfki.de/skincare/classify.html>.

## Requirements

A system with Python 3.6 or 3.7 already installed.

The framework is build on the top of Keras and TensofFlow libraries.

## Installation

Create a Python3 environment and install the package from the wheel archive

### From source

Install the needed libraries through the `pip` tool. Preferably, to avoid clashes, use a dedicated python environment.

```bash
git clone https://github.com/DFKI-Interactive-Machine-Learning/TIML.git
cd TIML
python3 -m venv p3env-timl
source p3env-timl/bin/activate

pip install -r requirements.txt
```

### From a release

```bash
cd path/to/TIML
python3 -m venv timl-p3env
source timl-p3env/bin/activate

pip install -U timl-x.y.z-py3-none-any.whl
```

## Files

IN PROGRESS...

A TIML release is distributed as single archive file, e.g.: `TIML-release-x.y.z.zip`.

Unpack the archive in an _empty_ directory.
You will see the following files:

* `README.md`
  * This file.
* `timl-x.y.z-py3-none-any.whl`
  * The installable python package.
* `server_config.json`
    * The configuration file for the server. You need to edit it only if you update the models.
* `Docs/`
    * Project documentation.
* `html/` Some web pages for testing purpose
  * `classify.html` Desktop-friendly test page
  * `classifytouch.html` Tablet-friendly test page
  * ... and other support files

## Documentation

Look into the `Docs` folder for dedicated docs:

* A manual for [USERS](Docs/USERS.md), who want to learn how to use this toolkit.
  * There is also a folder with Examples. E.g., [Example01](Examples/Example01/README.md).
* A manual for [DEVELOPERS](Docs/DEVELOPERS.md), who might want to extend this toolkit.
* A list of calls available in the [REST-API](Docs/REST-API.md).

## Running the Web server

The http REST interface is implemented using Flask.
To run the server from a terminal:

```bash
cd path/to/TIML
source timl-p3env/bin/activate
export FLASK_APP=timl.networking.__main__.py
python -m flask run
```

By default, the server takes connections on port 5000.
In order to specify a different port:

    python -m flask run --host=0.0.0.0 --port=80


To test the system open one of the following links in a browser:

* `http://127.0.0.1:5000/html/classify.html` Use the system through the desktop interface.
* `http://127.0.0.1:5000/html/classifytouch.html` Use the system through the desktop interface.
* `http://127.0.0.1:5000/html/evaluate.html` Evaluate classification results.


## Working with the network REST interface

The same port, TIML offers a REST API offering classification as remote service.
For example:

* `http://127.0.0.1:5000/model_info` returns info about the loaded classifiers.
* `http://127.0.0.1:5000/classify/binary` performs the actual classification. This must be a POST, providing the image.

See the document [REST-API](Docs/REST-API.md) for more information on how to invoke the REST API.

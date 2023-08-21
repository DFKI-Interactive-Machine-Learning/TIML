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


## Working with the network REST interface

TIML integrates a web-server that can be used to provide access to a REST-API or to load an explorative web inteface.

For example:

* `http://127.0.0.1:5000/info` returns info about the loaded classifiers.
   See the document [REST-API](Docs/REST-API.md) for more information on how to invoke the REST API.

* `http://127.0.0.1:5000/classify.html` load an interactive page for exploring classification results.

## Publications

To cite this work, please use the following BibTex entry (from https://dl.acm.org/doi/10.1145/3459926.3464753)

```
@inproceedings{10.1145/3459926.3464753,
author = {Nunnari, Fabrizio and Sonntag, Daniel},
title = {A Software Toolbox for Deploying Deep Learning Decision Support Systems with XAI Capabilities},
year = {2021},
isbn = {9781450384490},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459926.3464753},
doi = {10.1145/3459926.3464753},
abstract = {We describe the software architecture of a toolbox of reusable components for the configuration of convolutional neural networks (CNNs) for classification and labeling problems. The toolbox architecture has been designed to maximize the reuse of established algorithms and to include domain experts in the development and evaluation process across different projects and challenges. In addition, we implemented easy-to-edit input formats and modules for XAI (eXplainable AI) through visual inspection capabilities. The toolbox is available for the research community to implement applied artificial intelligence projects.},
booktitle = {Companion of the 2021 ACM SIGCHI Symposium on Engineering Interactive Computing Systems},
pages = {44–49},
numpages = {6},
keywords = {design patterns, explainable AI., object-oriented architecture, convolutional neural networks, Software toolbox, deep learning},
location = {Virtual Event, Netherlands},
series = {EICS '21}
}
```

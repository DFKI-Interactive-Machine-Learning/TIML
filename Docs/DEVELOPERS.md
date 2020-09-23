# TIML - Developers manual

## Development environment

The suggested IDE for developing TIML is PyCharm.

Prepare your own Python (v3.6 or 3.7) environment and activate it.

`pip install -r requirements.txt`

If you want a fresh version with all up-to-date packages, consider teh following list, and update packages version accordingly.

```bash
pip install pytest==4.3.0
pip install pandas==0.25.3
pip install tensorflow==1.13.1
pip install keras==2.2.4
pip install pillow==5.4.1
pip install matplotlib==3.0.3
pip install Flask==1.0.2
pip install requests==2.21.0

# To be removed. Used only to convert a saliency map to heatmap.
pip install opencv-python==4.1.1.26

# To be removed. Use only by RISE to rescale an image
pip install scikit-image==0.15.0
```

For plotting, you need to install the right back-end of matplot lib.
On Ubuntu 18.04 we needed:

```bash
sudo apt-get install python3-tk
```

NOTE! In the shared requirements file we use tensorflow, CPU-only version.
There are too many HW-related issues when trying to install the GPU version for everyone.
If you want to use the GPU version of tensor flow, please install it on your local machine only, after fulfilling all machine specific requirements (drivers, ...).

```bash
pip install -r requirements
pip uninstall tensorflow
pip install tensorflow-gpu==x.y.z
```


## Code Organization

```txt
Docs                    -- Documentation
Examples                -- Example of use of the TIML framework
LICENSE                 -- The open source license
README.md               -- This file
Scripts/                -- Useful scripts to manage datasets, and more...
html/                   -- Web pages for visual inspection
requirements.txt        -- The requirements for the development Python environment
setup.py                -- Installation script to create a wheel package of TIML
timl/                   -- The main Python package
    classification/     -- The classification tools (training, prediction and tests)
    common/             -- Common, general purpose, utilities.
    data/               -- Static data, embedded in the package, mainly for testing.
    networking/         -- Web server code and REST-API.
    xai/                -- eXplainable AI algorithms. E.g., Grad-CAM and RISE.
vis/                    -- A dump of the _Keras Visualization Toolkit_ (<https://github.com/raghakot/keras-vis>), mainly needed for the GrtadCAM implementation.
```

## Running modules

All the command lines of this project are implemented as "executable module",
meaning that some module directories have a `__main__.py` method which allows
for command line invocation. E.g.:

```bash
cd TIML/
# Now we are in the directory containing the `timl` package
python -m timl.train
Using TensorFlow backend.
usage: __main__.py [-h] [--img-dir IMG_DIR] [--cuda-gpu CUDA_GPU]
                   <input_table.csv>
__main__.py: error: the following arguments are required: <input_table.csv>
```
 

## Packaging

Use the setup tools to distribute the package:

```bash
cd Classifiers
python setup.py sdist
python setup.py bdist_wheel
```

And check the `dist` directory.

```
ls -l dist/
-rw-r--r-- 1 fnunnari fnunnari 32217 Apr 18 12:07 skincare_dfki-0.0.1-py3-none-any.whl
-rw-r--r-- 1 fnunnari fnunnari 21418 Apr 18 11:50 skincare-dfki-0.0.1.tar.gz
```

To install a package:

```bash
pip install -U ../GitLabSkinCare/Classifiers/dist/skincare_dfki-0.0.1-py3-none-any.whl
```

## Package resources

TIML stores some binary resources under

    timl.data

The `data` folder must be a package (`__init.py__` is in it) and is packed into the distributable wheel.
To acces the resources use the `pkg_resources` package. E.g., the `resource_listdir` and `resource_filename` methods. E.g.:

    images = pkg_resources.resource_listdir("timl.data", "sample_images")
  
## Test units

TIML uses [Pytest](https://docs.pytest.org/) for writing and running test units.

Tests are packed per-module, under a test directory. E.g.:

```
timl/
    classification/
        test/
            test_models.py
            ...
```

To run all the tests, from the console:

    cd TIML
    pytest

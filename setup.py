import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timl",
    version="0.1.2",
    author="Fabrizio Nunnari",
    author_email="fabrizio.nunnari@dfki.de",
    description="The Skincare project aims at providing tools for the analysis and processing of skin lesions.",
    long_description=long_description,
    url="https://ai-in-medicine.dfki.de/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pytest==4.3.0',
        'pandas==0.25.3',
        'tensorflow==1.13.1',
        'keras==2.2.4',
        'Pillow==5.4.1',
        'matplotlib==3.0.3',
        'Flask==1.0.2',
        'requests==2.21.0',
        'opencv-python==4.1.1.26',
        'scikit-image==0.15.0'
    ],

    # See https://setuptools.readthedocs.io/en/latest/setuptools.html#including-data-files
    package_data={'timl.data': '*'}

)

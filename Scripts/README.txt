Some scripts need the timl module in the search path.
If needed, set the search path like in this example:

```
cd TIML/Scripts/
PYTHONPATH=.. python ClassifyImage.py skincare_train_output-20190715-172428-LES_UNK-450/0-keras_model-20190715-172428.h5 sample_images/ISIC_0000000.jpeg
```


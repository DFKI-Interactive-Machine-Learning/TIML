all: train predict inspect


train: sample_train_output

sample_train_output:
	PYTHONPATH=../.. python -m timl.classification.train --input-table=sample_train_input_1.csv --img-dir=../../timl/data/ISIC2019/images --out-dir=$@
	touch $@


predict: sample_predictions.csv

sample_predictions.csv:
	PYTHONPATH=../.. python -m timl.classification.predict --keras-model=sample_train_output/0-keras_model.h5 --config-table=sample_train_output/0-automation_result.csv --test-dataframe=../../timl/data/ISIC2019/ISIC_2019_Training_GroundTruth_meta-test.csv --img-dir=../../timl/data/ISIC2019/images --output-csv=$@


inspect: sample_inspection

sample_inspection:
	PYTHONPATH=../.. python -m timl.classification.inspect --test-csv=../../timl/data/ISIC2019/ISIC_2019_Training_GroundTruth_meta-test.csv --predictions-csv=sample_predictions.csv --overwrite --out-dir=$@
	touch $@

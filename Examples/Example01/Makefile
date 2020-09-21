all: train predict inspect

TRAIN_INPUT=sample_train_input_1.csv
TRAIN_OUTPUT=sample_train_output
PREDICTION_OUTPUT=sample_predictions.csv
INSPECTION_OUTPUT=sample_inspection

clean:
	rm -rf $(TRAIN_OUTPUT)
	rm $(PREDICTION_OUTPUT)
	rm -rf $(INSPECTION_OUTPUT)


train: $(TRAIN_OUTPUT)

$(TRAIN_OUTPUT): $(TRAIN_INPUT)
	PYTHONPATH=../.. python -m timl.classification.train --input-table=sample_train_input_1.csv --img-dir=../../timl/data/ISIC2019/images --out-dir=$@
	touch $@


predict: $(PREDICTION_OUTPUT)

$(PREDICTION_OUTPUT): $(TRAIN_OUTPUT)
	PYTHONPATH=../.. python -m timl.classification.predict --keras-model=$(TRAIN_OUTPUT)/0-keras_model.h5 --config-table=$(TRAIN_OUTPUT)/0-automation_result.csv --test-dataframe=../../timl/data/ISIC2019/ISIC_2019_Training_GroundTruth_meta-test.csv --img-dir=../../timl/data/ISIC2019/images --output-csv=$@


inspect: $(INSPECTION_OUTPUT)

$(INSPECTION_OUTPUT): $(PREDICTION_OUTPUT)
	PYTHONPATH=../.. python -m timl.classification.inspect --test-csv=../../timl/data/ISIC2019/ISIC_2019_Training_GroundTruth_meta-test.csv --predictions-csv=$(PREDICTION_OUTPUT) --overwrite --out-dir=$@
	touch $@

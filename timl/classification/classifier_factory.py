from timl.classification.classifier import Classifier


def make_classifier(train_method: str, image_size: int, n_classes: int) -> Classifier:
    """This is he factory method to instantiate a Classifier sub-class.
    See: https://en.wikipedia.org/wiki/Factory_method_pattern"""

    if train_method == "VGG16" or train_method == "VGG16flat":
        from timl.classification.vgg16.classifiers import VGG16FlatClassifier
        classifier = VGG16FlatClassifier(input_dim=(image_size, image_size), n_classes=n_classes)
    elif train_method == "RESNET50":
        from timl.classification.resnet import ResNet50Classifier
        from keras.optimizers import SGD
        sgd = SGD(lr=1e-5, decay=1e-4, momentum=0.9, nesterov=True)
        classifier = ResNet50Classifier(input_dim=(image_size, image_size), n_classes=n_classes, optimizer=sgd)

    #
    # Architectures for Multi-label problems

    elif train_method == "VGG16-fc2k-ml":
        from timl.classification.vgg16.classifiers import VGG16ClassifierML
        from keras.optimizers import Nadam
        classifier = VGG16ClassifierML(input_dim=(image_size, image_size),
                                       n_classes=n_classes,
                                       optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                       fc_shape=[2048, 2048])
    elif train_method == "VGG16-fc4k-ml":
        from timl.classification.vgg16.classifiers import VGG16ClassifierML
        from keras.optimizers import Nadam
        classifier = VGG16ClassifierML(input_dim=(image_size, image_size),
                                       n_classes=n_classes,
                                       optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                       fc_shape=[4096, 4096])
    elif train_method == "VGG16-fc4k-do75-ml":
        from timl.classification.vgg16.classifiers import VGG16ClassifierML
        from keras.optimizers import Nadam
        classifier = VGG16ClassifierML(input_dim=(image_size, image_size),
                                       n_classes=n_classes,
                                       optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                       fc_shape=[4096, 4096],
                                       dropouts=[0.75, 0.75])
    elif train_method == "VGG16-fc2k-do75-ml":
        from timl.classification.vgg16.classifiers import VGG16ClassifierML
        from keras.optimizers import Nadam
        classifier = VGG16ClassifierML(input_dim=(image_size, image_size),
                                       n_classes=n_classes,
                                       optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                       fc_shape=[2048, 2048],
                                       dropouts=[0.75, 0.75])
    elif train_method == "VGG16-1Xfc2k-do75-ml":
        from timl.classification.vgg16.classifiers import VGG16ClassifierML
        from keras.optimizers import Nadam
        classifier = VGG16ClassifierML(input_dim=(image_size, image_size),
                                       n_classes=n_classes,
                                       optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                       fc_shape=[2048],
                                       dropouts=[0.75],
                                       fc_initializer=['glorot_uniform'] * 2)
    elif train_method == "VGG16-3Xfc4k-ml":
        from timl.classification.vgg16.classifiers import VGG16ClassifierML
        from keras.optimizers import Nadam
        classifier = VGG16ClassifierML(input_dim=(image_size, image_size),
                                       n_classes=n_classes,
                                       optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                       fc_shape=[4096] * 3,
                                       dropouts=[0.5] * 3,
                                       fc_initializer=['glorot_uniform'] * 4
                                       )
    elif train_method == "VGG16-fc8k-ml":
        from timl.classification.vgg16.classifiers import VGG16ClassifierML
        from keras.optimizers import Nadam
        classifier = VGG16ClassifierML(input_dim=(image_size, image_size),
                                       n_classes=n_classes,
                                       optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                       fc_shape=[8192, 8192])
    elif train_method == "VGG16-fc4k-ml-lossf1":
        from timl.classification.vgg16.classifiers import VGG16ClassifierML
        from keras.optimizers import Nadam
        classifier = VGG16ClassifierML(input_dim=(image_size, image_size),
                                       n_classes=n_classes,
                                       optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                       fc_shape=[4096, 4096])
    elif train_method == "RESNET50-fc4k-ml":
        from timl.classification.resnet import ResNet50ClassifierML
        from keras.optimizers import Nadam
        from timl.classification.metrics import mean_f1_score
        classifier = ResNet50ClassifierML(input_dim=(image_size, image_size),
                                          n_classses=n_classes,
                                          optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                          fc_shape=(4096, 4096),
                                          loss_function='binary_crossentropy',
                                          activation='sigmoid',
                                          metric_list=['accuracy', mean_f1_score])
    elif train_method == "DenseNet169-fc2k-ml":
        from timl.classification.densenet import DenseNet169ClassifierML
        from keras.optimizers import Nadam
        from timl.classification.metrics import mean_f1_score
        classifier = DenseNet169ClassifierML(input_dim=(image_size, image_size),
                                             n_classses=n_classes,
                                             optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                             fc_shape=(2048, 2048),
                                             loss_function='binary_crossentropy',
                                             activation='sigmoid',
                                             metric_list=['accuracy', mean_f1_score])
    elif train_method == "DenseNet169-fc4k-ml":
        from timl.classification.densenet import DenseNet169ClassifierML
        from keras.optimizers import Nadam
        from timl.classification.metrics import mean_f1_score
        classifier = DenseNet169ClassifierML(input_dim=(image_size, image_size),
                                             n_classses=n_classes,
                                             optimizer=Nadam(lr=1e-5, schedule_decay=0.9),
                                             fc_shape=(4096, 4096),
                                             loss_function='binary_crossentropy',
                                             activation='sigmoid',
                                             metric_list=['accuracy', mean_f1_score])

    #
    # If no configuration matches
    else:
        raise Exception("Training method {} not recognized.".format(train_method))

    assert classifier is not None

    return classifier

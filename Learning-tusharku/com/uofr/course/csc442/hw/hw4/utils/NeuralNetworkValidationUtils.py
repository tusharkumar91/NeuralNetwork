from com.uofr.course.csc442.hw.hw4.models.NeuralNetwork import NeuralNetwork
from com.uofr.course.csc442.hw.hw4.dataloader.DataLoader import DataLoader
import numpy as np
import time


def validate_arch(train_data,
                  validate_data,
                  archs_to_validate,
                  target_attribute,
                  attribute_file):
    r"""
    Method to validate the appropriate batch size that should be valid
    for a particular dataset.
    :param train_data: data on which the model needs to be trained
    :param validate_data: data on which different depths need to be validated
    :param archs_to_validate: architectures which we will try validating
    :param target_attribute: attribute which needs to be predicted
    :param attribute_file: file which contains the details of attribute
    :return: None
    """
    validation_accuracy = {}
    best_arch = []
    best_arch_validation_accuracy = 0.0
    time_arch = {}
    np.random.seed(0)
    for arch in archs_to_validate:
        print("="*50)
        print("Learning Neuralnet for architecture : {}".format(arch))
        start_time = time.time()
        nnet = NeuralNetwork(attribute_file=attribute_file,
                             train_data=train_data,
                             test_data=validate_data,
                             target_attribute=target_attribute,
                             layer_sizes=arch)

        nnet.train(epochs=75, learning_rate=0.05, batch_size=1, validation_interval=10)
        pred = nnet.predict()
        accuracy = nnet.get_accuracy(predictions=pred)
        print("Validation Accuracy for arch {} : {}".format(arch, accuracy))
        validation_accuracy[str(arch)] = accuracy
        time_arch[str(arch)] = time.time() - start_time
        if accuracy > best_arch_validation_accuracy:
            best_arch = arch
            best_arch_validation_accuracy = accuracy
        print("=" * 50)

    print("Best arch found during validation : {} with accuracy {}".format(best_arch, best_arch_validation_accuracy))
    accuracies = []
    batch_sizes = []
    timestamps = []
    for key in validation_accuracy.keys():
        batch_sizes.append(key)
        accuracies.append(validation_accuracy[key])
        timestamps.append(time_arch[key])
    print("Architectures", batch_sizes)
    print("Accuracy", accuracies)
    print("Time", timestamps)


def validate_batch_size(train_data, validate_data, batch_sizes_to_validate, target_attribute, attribute_file):
    r"""
    Method to validate the appropriate batch size that should be valid
    for a particular dataset.
    :param train_data: data on which the model needs to be trained
    :param validate_data: data on which different depths need to be validated
    :param batch_sizes_to_validate: batch sizes which we will try validating
    :param target_attribute: attribute which needs to be predicted
    :param attribute_file: file which contains the details of attribute
    :return: None
    """
    validation_accuracy = {}
    best_batch_size = 0
    best_batch_size_validation_accuracy = 0.0
    time_batch_size = {}
    np.random.seed(0)
    for batch_size in batch_sizes_to_validate:
        print("="*50)
        print("Learning Neuralnet for batch size : {}".format(batch_size))
        start_time = time.time()
        nnet = NeuralNetwork(attribute_file=attribute_file,
                             train_data=train_data,
                             test_data=validate_data,
                             target_attribute=target_attribute,
                             layer_sizes=[126, 100, 3])

        nnet.train(epochs=75, learning_rate=0.05, batch_size=batch_size, validation_interval=10)
        pred = nnet.predict()
        accuracy = nnet.get_accuracy(predictions=pred)
        print("Validation Accuracy for batch_size {} : {}".format(batch_size, accuracy))
        validation_accuracy[batch_size] = accuracy
        time_batch_size[batch_size] = time.time() - start_time
        if accuracy > best_batch_size_validation_accuracy:
            best_batch_size = batch_size
            best_batch_size_validation_accuracy = accuracy
        print("=" * 50)

    print("Best batch_size found during validation : {} with accuracy {}".format(best_batch_size,
                                                                                 best_batch_size_validation_accuracy))
    accuracies = []
    batch_sizes = []
    timestamps = []
    for key in validation_accuracy.keys():
        batch_sizes.append(key)
        accuracies.append(validation_accuracy[key])
        timestamps.append(time_batch_size[key])
    print("Batch Sizes", batch_sizes)
    print("Accuracy", accuracies)
    print("Time", timestamps)


def validate_learning_rate(train_data,
                           validate_data,
                           lr_to_validate,
                           target_attribute,
                           attribute_file):
    r"""
    Method to validate the appropriate batch size that should be valid
    for a particular dataset.
    :param train_data: data on which the model needs to be trained
    :param validate_data: data on which different depths need to be validated
    :param lr_to_validate: learning rates which we will try validating
    :param target_attribute: attribute which needs to be predicted
    :param attribute_file: file which contains the details of attribute
    :return: None
    """
    validation_accuracy = {}
    best_lr = 0
    best_lr_validation_accuracy = 0.0
    time_lr = {}
    np.random.seed(0)
    for lr in lr_to_validate:
        print("="*50)
        print("Learning Neuralnet for learning rate : {}".format(lr))
        start_time = time.time()
        nnet = NeuralNetwork(attribute_file=attribute_file,
                             train_data=train_data,
                             test_data=validate_data,
                             target_attribute=target_attribute,
                             layer_sizes=[16, 10, 3])

        nnet.train(epochs=75, learning_rate=lr, batch_size=1, validation_interval=5)
        pred = nnet.predict()
        accuracy = nnet.get_accuracy(predictions=pred)
        validation_accuracy[lr] = accuracy
        time_lr[lr] = time.time() - start_time
        if accuracy > best_lr_validation_accuracy:
            best_lr = lr
            best_lr_validation_accuracy = accuracy
        print("=" * 50)

    print("Best learning rate found during validation : {} with accuracy {}".format(best_lr,
                                                                                    best_lr_validation_accuracy))
    accuracies = []
    lrs = []
    timestamps = []
    for key in validation_accuracy.keys():
        lrs.append(key)
        accuracies.append(validation_accuracy[key])
        timestamps.append(time_lr[key])
    print("Learning rate", lrs)
    print("Accuracy", accuracies)
    print("Time", timestamps)


def run_validation():
    print(np.version.version)
    exit(0)
    data_file = "../data/connect-4.data.txt"
    attribute_file = "../data/connect-4-desc.txt"
    attribute_file = "../data/iris-desc.txt"
    data_file = "../data/iris.data.discrete.txt"
    attribute_names, _ = DataLoader.load_attribute_data(attribute_file=attribute_file)
    train_data = DataLoader.load_data_from_file(data_file, attribute_names)
    train_data, test_data = DataLoader.train_test_or_validate_split(train_data)
    train_data, validation_data = DataLoader.train_test_or_validate_split(train_data, test_or_validate_ratio=0.25)
    #validate_batch_size(train_data=train_data, validate_data=validation_data, target_attribute="Class",
    #                    attribute_file=attribute_file,
    #                    batch_sizes_to_validate=[1, 50, 100, 500, 1000])
    validate_arch(train_data=train_data, validate_data=validation_data, target_attribute="Class",
     attribute_file=attribute_file,archs_to_validate=[[16, 1, 3], [16, 10, 3],[16, 50, 3],[16, 100, 3], [16, 1000, 3],
     [16, 10, 10, 3],[16, 100, 100, 3], [16, 10, 10, 10, 3], [16, 100, 100, 100, 3]])
    # validate_learning_rate(train_data=train_data, validate_data=validation_data, target_attribute="Class",
    #                    attribute_file=attribute_file, lr_to_validate=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])


if __name__ == '__main__':
    run_validation()
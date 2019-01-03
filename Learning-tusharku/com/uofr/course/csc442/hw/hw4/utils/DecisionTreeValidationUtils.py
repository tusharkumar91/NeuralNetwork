import pickle
from com.uofr.course.csc442.hw.hw4.models.DecisionTree import DecisionTree
from com.uofr.course.csc442.hw.hw4.dataloader.DataLoader import DataLoader
import numpy as np

def validate_depth_of_tree(train_data,
                           validate_data,
                           depths_to_validate,
                           target_attribute,
                           attribute_file):
    r"""
    Method to validate the depth of tree which should be appropriate
    for a particular dataset.
    :param train_data: data on which the model needs to be trained
    :param validate_data: data on which different depths need to be validated
    :param depths_to_validate: depths which we will try validating
    :param target_attribute: attribute which needs to be predicted
    :param attribute_file: file which contains the details of attribute
    :return: None
    """
    validation_accuracy = {}
    best_depth = 0
    best_depth_validation_accuracy = 0.0
    for depth in depths_to_validate:
        print("="*50)
        print("Learning Tree for depth : {}".format(depth))
        decision_tree = DecisionTree(attribute_file=attribute_file,
                                     train_data=train_data,
                                     test_data=validate_data,
                                     target_attribute=target_attribute,
                                     max_depth=depth)

        decision_tree.train()
        pred = decision_tree.predict()

        # 6. Get Performance Metrics
        accuracy = decision_tree.get_accuracy(predictions=pred)
        print("Validation Accuracy for depth {} : {}".format(depth, accuracy))
        validation_accuracy[depth] = accuracy
        if accuracy > best_depth_validation_accuracy:
            best_depth = depth
            best_depth_validation_accuracy = accuracy
        print("=" * 50)
        print()

    print("Best depth found during validation : {} with accuracy {}".format(best_depth, best_depth_validation_accuracy))
    accuracies = []
    depths = []
    for key in validation_accuracy.keys():
        depths.append(key)
        accuracies.append(validation_accuracy[key])
    print(depths)
    print(accuracies)
    return depths, accuracies


def validate_splitting_criteria(train_data,
                                validate_data,
                                splitting_criterion,
                                target_attribute,
                                attribute_file,
                                depth):
    r"""
    Method to validate the depth of tree which should be appropriate
    for a particular dataset.
    :param train_data: data on which the model needs to be trained
    :param validate_data: data on which different depths need to be validated
    :param target_attribute: attribute which needs to be predicted
    :param attribute_file: file which contains the details of attribute
    :param depth : max depth for which the tree will be created
    :param splitting_criterion : the different criterias to be validated
    :return:
    """
    validation_accuracy = {}
    best_criteria = ""
    best_criteria_validation_accuracy = 0.0
    for criteria in splitting_criterion:
        print("="*50)
        print("Learning Tree with criteria : {}".format(criteria))
        decision_tree = DecisionTree(attribute_file=attribute_file,
                                     train_data=train_data,
                                     test_data=validate_data,
                                     target_attribute=target_attribute,
                                     max_depth=depth,
                                     splitting_criteria=criteria)

        decision_tree.train()
        pred = decision_tree.predict()

        # 6. Get Performance Metrics
        accuracy = decision_tree.get_accuracy(predictions=pred)
        print("Validation Accuracy for criteria {} : {}".format(criteria, accuracy))
        validation_accuracy[criteria] = accuracy
        if accuracy > best_criteria_validation_accuracy:
            best_criteria = criteria
            best_criteria_validation_accuracy = accuracy
        print("=" * 50)
        print()

    print("Best criteria found during validation : {} with accuracy {}".
          format(best_criteria, best_criteria_validation_accuracy))
    accuracies = []
    criterion = []
    for key in validation_accuracy.keys():
        criterion.append(key)
        accuracies.append(validation_accuracy[key])
    print(criterion)
    print(accuracies)
    return criterion, accuracies


def run_validation():
    attribute_file = "../data/iris-desc.txt"
    data_file = "../data/iris.data.discrete.txt"
    attribute_names, _ = DataLoader.load_attribute_data(attribute_file=attribute_file)
    accuracy = np.zeros(2)
    for i in range(1000):
        #with open("../train/connect-4.data", "rb") as f:
        #    train_data = pickle.load(f)
        train_data = DataLoader.load_data_from_file(data_file, attribute_names)
        train_data, test_data = DataLoader.train_test_or_validate_split(train_data)
        train_data, validation_data = DataLoader.train_test_or_validate_split(train_data, test_or_validate_ratio=0.25)
        _, accuracies = validate_splitting_criteria(train_data=train_data, validate_data=validation_data,
                                target_attribute="Class", attribute_file=attribute_file,
                                splitting_criterion=["entropy", "gini"], depth=3)
        accuracy = np.add(accuracy, np.array(accuracies))
        print("Finished iteration {}".format(i))
    print(accuracy/1000)
    # validate_splitting_criteria(train_data=train_data, validate_data=validation_data,
    #                            target_attribute="Class", attribute_file=attribute_file,
    #                            splitting_criterion=["entropy", "gini"], depth=10)


if __name__ == '__main__':
    run_validation()
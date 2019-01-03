r"""
Main script to learn and test the ml models
"""

import argparse
import numpy as np
from com.uofr.course.csc442.hw.hw4.models.DecisionTree import DecisionTree
from com.uofr.course.csc442.hw.hw4.dataloader.DataLoader import DataLoader
from com.uofr.course.csc442.hw.hw4.models.NeuralNetwork import NeuralNetwork
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CSC442HW4 Learning Project')

    # Program related settings

    parser.add_argument('--dataset', type=str, default="iris",
                        help='dataset to use for learning')
    parser.add_argument('--train_test_split', default=0.2, type=float,
                        help='Split to be used for training and testing')
    parser.add_argument('--model', default="dtree", type=str,
                        help='Model to be used for learning')

    # Training Neural Net procedure settings

    parser.add_argument('--log_interval', type=int, default=5,
                        help='report interval after N epochs')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--lr', '--learning-rate', default=5E-2, type=float,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=11,
                        help='random seed')

    # Training Decision Tree procedure settings
    parser.add_argument('--max_depth', type=int, default=None,
                        help='depth till which tree should be grown')
    parser.add_argument('--splitting_criteria', default="entropy", type=str,
                        help='splitting criteria for tree')
    parser.add_argument('--print_tree', default=False, action="store_true",
                        help='whether to print tree or not')

    args = parser.parse_args()

    args_dict = vars(args)

    print('\n\nArgument list to program\n\n')

    print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                     for arg in args_dict]))

    np.random.seed(args.seed)

    dataset_path = {"connect-4" : "com/uofr/course/csc442/hw/hw4/data/connect-4.data.txt",
                    "iris" : "com/uofr/course/csc442/hw/hw4/data/iris.data.discrete.txt",
                    "aima-restaurant" : "com/uofr/course/csc442/hw/hw4/data/AIMA_Restaurant-data.txt"}.get(args.dataset)

    dataset_attribute_path = {"connect-4": "com/uofr/course/csc442/hw/hw4/data/connect-4-desc.txt",
                    "iris": "com/uofr/course/csc442/hw/hw4/data/iris-desc.txt",
                    "aima-restaurant": "com/uofr/course/csc442/hw/hw4/data/AIMA_Restaurant-desc.txt"}.get(args.dataset)

    target_attribute = {"connect-4": "Class",
                    "iris": "Class",
                    "aima-restaurant": "WillWait"}.get(args.dataset)

    if dataset_attribute_path is None or dataset_path is None:
        raise ValueError("dataset can only be connect-4 or iris or aima-restaurant but was provided " + args.dataset)

    attribute_names, attribute_domains = DataLoader.load_attribute_data(dataset_attribute_path)

    data = DataLoader.load_data_from_file(data_file=dataset_path, attribute_names=attribute_names)

    train_data, test_data = DataLoader.train_test_or_validate_split(data, test_or_validate_ratio=args.train_test_split)

    if args.model == "dtree":
        decision_tree = DecisionTree(attribute_file=dataset_attribute_path,
                                     train_data=train_data,
                                     test_data=test_data,
                                     target_attribute=target_attribute,
                                     max_depth=args.max_depth,
                                     splitting_criteria=args.splitting_criteria)

        print("\nLearning the Tree\n\n")
        learning_start_time = time.time()

        decision_tree.train()

        print("Learning took {} seconds".format(str(time.time() - learning_start_time)))

        print("Depth of tree learned is {}".format(DecisionTree.find_depth_tree(decision_tree.classifier)))

        if args.print_tree is True:
            decision_tree.print_decision_tree(decision_tree.classifier, indent="", default_indent="\t\t")

        if len(test_data) > 0:
            pred = decision_tree.predict()
            accuracy = decision_tree.get_accuracy(predictions=pred)
            print("\n=======================================\n")
            print("Accuracy of Decision tree classifier : {}".format(accuracy))
            print("\n=======================================\n")


    elif args.model == "nnet":

        layer_sizes = {"connect-4": [126, 100, 3],
                       "iris": [16, 10, 3],
                       "aima-restaurant": [26, 10, 2]}.get(args.dataset)

        nnet = NeuralNetwork(attribute_file=dataset_attribute_path,
                             train_data=train_data,
                             test_data=test_data,
                             target_attribute=target_attribute,
                             layer_sizes=layer_sizes)

        print("\nLearning the Neural Network\n\n")
        learning_start_time = time.time()
        nnet.train(batch_size=args.batch_size, learning_rate=args.lr, epochs=args.epochs, validation_interval=args.log_interval)
        print("Learning took {} seconds".format(str(time.time() - learning_start_time)))
        pred = nnet.predict()
        accuracy = nnet.get_accuracy(predictions=pred)
        print("\n=======================================\n")
        print("Accuracy of Neural Net : {}".format(accuracy))
        print("\n=======================================\n")
    else:
        raise ValueError("model can only be dtree or nnet but was provided " + args.model)



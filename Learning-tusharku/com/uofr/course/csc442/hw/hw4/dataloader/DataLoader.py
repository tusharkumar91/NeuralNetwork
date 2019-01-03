import numpy as np


class DataLoader:
    r"""
    Class to act as a utility to help with all the data loading
    functionality for the ML models

    Assumptions are that the attribute file format will have the following syntax
    ======================
    attribute_name_1:possible_value1/possible_value2/...../possible_valueN
    attribute_name_2:possible_value1/possible_value2/...../possible_valueN
    ....
    ======================

    The data file should have following syntax
    ======================
    row1_attribute_value1,row1_attribute_value2,...,row1_attribute_valueN
    row2_attribute_value1,row2_attribute_value2,...,row2_attribute_valueN
    ======================
    """
    @staticmethod
    def load_attribute_data(attribute_file):
        attribute_names = []
        attribute_domains = {}
        with open(attribute_file, 'r') as f:
            for line in f:
                attr_name = line.split(":")[0]
                attribute_names.append(attr_name)
                attribute_values = line.split(":")[1]
                attribute_values = attribute_values.strip()
                attribute_domains[attr_name] = set([value for value in attribute_values.split("/")])
        return attribute_names, attribute_domains

    @staticmethod
    def load_data_from_file(data_file, attribute_names):
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                attr_values = line.split(",")
                data_row = {}
                for idx, attr_value in enumerate(attr_values):
                    data_row[attribute_names[idx]] = attr_value.rstrip("\n\r")
                data.append(data_row)
        return data

    @staticmethod
    def train_test_or_validate_split(data, test_or_validate_ratio=0.2, shuffle=True):
        r"""
        Function to partition the dataSet into train and testData.
        testRatio -> ratio of test examples:dataSet size. Default = 0.2
        shouldShuffle -> whether the data needs to be shuffled or not. Default = false
        :param data: data which needs to be split
        :param test_or_validate_ratio: ratio of validation or testing
        :param shuffle: if the rows should be shuffled before splittiing
        :return: train_data and test/validate_data
        """
        N = len(data)
        if shuffle:
            np.random.shuffle(data)
        train_size = int(N * (1 - test_or_validate_ratio))
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data
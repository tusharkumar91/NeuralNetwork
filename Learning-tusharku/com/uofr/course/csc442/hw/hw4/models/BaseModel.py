
from com.uofr.course.csc442.hw.hw4.dataloader.DataLoader import DataLoader


class BaseModel:
    r""" Base class for encapsulating all the common
    process done for every ML model that subclasses this class.
    Typically the common behaviors are to get attribute information
    like the different names of attributes and their domains.
    It also sets up the training and testing data
    depending on if the data was actually provided as a processed version
    or it needs to be extracted from a file .

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

    def __init__(self, attribute_file, target_attribute, train_data_file=None,
                 test_data_file=None, train_data=None, test_data=None):
        self.attribute_file = attribute_file
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.train_data = train_data
        self.test_data = test_data
        self.target_attribute = target_attribute
        self.attribute_names = []
        self.attribute_domains = {}

    def get_attribute_data(self):
        r"""
        Loads the attribute details from the attribute file
        that was provided
        :return:attribute names and domain values
        """
        self.attribute_names, self.attribute_domains = DataLoader.load_attribute_data(self.attribute_file)

    def get_train_data(self):
        if self.train_data is None and self.train_data_file is None:
                raise ValueError("Neither training data not training file provided")

        if self.train_data is None and self.train_data_file is not None:
            self.train_data = DataLoader.load_data_from_file(self.train_data_file, self.attribute_names)

    def get_test_data(self):
        if self.test_data is None and self.test_data_file is None:
            raise ValueError("Neither testing data not testing file provided")
        data = []
        if self.test_data is None and self.test_data_file is not None:
            with open(self.test_data_file, 'r') as f:
                for line in f:
                    attr_values = line.split(",")
                    data_row = {}
                    for idx, attr_value in enumerate(attr_values):
                        data_row[self.attribute_names[idx]] = attr_value.rstrip("\n\r")
                    data.append(data_row)
                self.test_data = data

    def get_accuracy(self, predictions):
        r"""
        Function to calculate the accuracy by comparing the predictions
        with the value of target attribute for the rows in test data
        :param predictions: predicted values of target attribute
        :return: Accuracy of the model
        """
        error_count = 0
        for index in range(len(self.test_data)):
            if self.test_data[index][self.target_attribute] != predictions[index]:
                error_count += 1
        return (1 - (float(error_count) / len(self.test_data))) * 100

import numpy as np
from collections import defaultdict
from com.uofr.course.csc442.hw.hw4.models.BaseModel import BaseModel


class DecisionTree(BaseModel):
    r"""
     Class for DecisionTree Implementation:
     This is an implementation of Decision Trees
     modelling the ID3 implementation.
     I have also provided the splitting
     criterion as an input which can be gini/entropy.
    """
    def __init__(self, attribute_file, target_attribute, train_data_file=None,
                 test_data_file=None, train_data=None, test_data=None,
                 max_depth=None, splitting_criteria="entropy"):

        BaseModel.__init__(self, attribute_file=attribute_file, target_attribute=target_attribute,
                           train_data_file=train_data_file, test_data_file=test_data_file,
                           train_data=train_data, test_data=test_data)
        self.max_depth = max_depth
        self.splitting_criteria = splitting_criteria
        self.classifier = None

    class DecisionNode:
        r"""
        Class to encapsulate a node in the tree that is being
        created as a classifier.
        Every node will either have a label associated with it
        if it is a leaf node OR if will have a branch details
        associated with it where a branch is just the children
        forking out from this node.
        """
        def __init__(self, attribute_name=None, branches=None, label=None):
            self.attribute_name = attribute_name
            self.branches = branches
            self.label = label

    def get_entropy(self, rows):
        r"""
        Function to getEntropy of the set of rows .
        Entropy Formula = Sum over i (-p(i)*log(p(i)))
        Here p(i) represents the probability that element will
        belong to ith class if its picked at random from the rows subset.
        Example - Suppose there are 10 rows 6 with one labelValue and 4 with another .
            Then entropy = -0.6*log(0.6) - 0.4*log(0.4)

        :param rows: the tuples in the dataset
        :return: entropy value
        """
        label_count = defaultdict(int)
        total_count = 0
        for row in rows:
            label = row[self.target_attribute]
            label_count[label] += 1
            total_count += 1
        return sum([-(float(label_count[label]) /
                      total_count) * np.log2(float(label_count[label]) / total_count)
                    for label in label_count.keys()])

    def get_gini(self, rows):
        r"""
        Function to get gini impurity of the set of rows .
        Gini Formula = 1 - Sum over i (p(i)^2)
        Here p(i) represents the probability that element will
        belong to ith class if its picked at random from the rows subset.
        Example - Suppose there are 10 rows 6 with one labelValue and 4 with another .
            Then gini impurity = 1 - (0.6*0.6 + 0.4*0.4)
        :param rows: the tuples in the dataset
        :return: gini impurity value
        """
        label_count = defaultdict(int)
        total_count = 0
        for row in rows:
            label = row[self.target_attribute]
            label_count[label] += 1
            total_count += 1
        return 1 - sum([np.square(float(label_count[label])/total_count) for label in label_count.keys()])

    @staticmethod
    def perform_partition(rows, partition_attribute):
        r"""
        Function to partition the rows at a decisionNode.
        This will return a dictionary where keys are
        the attribute value of the column based on which
        partition is happening whereas value
        is the list of rows having that particular
        attribute value
        :param rows: the tuples in the dataset
        :param partition_attribute: the attribute based on values of which
            partitions will be created.
        :return: the partitioned rows
        """
        data_partitions = defaultdict(list)
        for row in rows:
            data_partitions[row[partition_attribute]].append(row)
        return data_partitions

    @staticmethod
    def get_count_by_attribute_value(rows, attribute_name):
        r"""
        Function to return the unique counts of each
        of the different possible values for
        attribute_name in the dataSet
        :param rows: the tuples in the dataset
        :param attribute_name: the attribute based on values of which the counts are measured
        :return: dictionary of attribute value to count mapping
        """
        results = defaultdict(lambda: 0)
        for row in rows:
            r = row[attribute_name]
            results[r] += 1
        return results

    @staticmethod
    def get_max_value_in_dictionary(mapping):
        max_value = max(mapping.values())
        choice = np.random.choice([key for key, value in mapping.items() if value == max_value])
        return choice

    def train(self):
        r"""
        Method to train the decision tree classifier.
        """
        # 1. Extracting details of attributes

        self.get_attribute_data()
        if self.train_data is None and self.train_data_file is None:
                raise ValueError("Neither training data not training file provided")

        self.get_train_data()
        self.classifier = self.build_tree(rows=self.train_data, attribute_list=self.attribute_names)

    def build_tree(self, rows, attribute_list, depth=1, parent_rows=None):
        r"""
        Building a tree
        1. Iterate through all attributes
        2. for each attribute iterate through all values to find one which has highest information gain
        3. use that to get the partitions sets and then for each of them work recursively
        This function will recursively create a decisionTree.
        At each call it will check for a attribute and the value of the attribute ,
        splitting on which will give the maximum  gain based on input splitting criteria
        And then it will recursively call the build_tree method to build the branches

        rows - rows of dataset which will be mapped to this node.
        target_attribute - attribute name which tree is supposed to learn to predict.
        attribute_list - List of different attributes which need to be checked for split
        attribute_domain - Domain of attribute values for all attributes
        parent_rows - Rows belonging to the decision node that is this nodes parent
        :param rows: the tuples in the dataset corresponding to below current node
        :param attribute_list: list of all attributes
        :param depth: current depth of tree
        :param parent_rows: rows attribute to parent of this node
        :return:
        """
        if len(rows) == 0:
            if parent_rows is not None:
                label_map = DecisionTree.get_count_by_attribute_value(parent_rows, self.target_attribute)
                return DecisionTree.DecisionNode(label=DecisionTree.get_max_value_in_dictionary(label_map))
            else:
                raise ValueError("Reached a decision node which had zero rows but was not"
                                 "provided with a parent node")
        if self.max_depth is not None and depth == self.max_depth:
            label_map = DecisionTree.get_count_by_attribute_value(rows, self.target_attribute)
            return DecisionTree.DecisionNode(label=DecisionTree.get_max_value_in_dictionary(label_map))

        try:
            splitting_func = {"entropy": self.get_entropy,
                              "gini": self.get_gini}.get(self.splitting_criteria)
        except KeyError:
            print("Program only supports entropy and gini as splitting criteria. Provided criteria was " +
                  self.splitting_criteria)
            raise ValueError("Incorrect  parameter value passed for splitting criteria")

        value_before_split = splitting_func(rows)

        if len(attribute_list) == 0 or value_before_split == 0:
            label_map = DecisionTree.get_count_by_attribute_value(rows, self.target_attribute)
            return DecisionTree.DecisionNode(label=DecisionTree.get_max_value_in_dictionary(label_map))

        if len(attribute_list) == 1 and attribute_list[0] == self.target_attribute:
            label_map = DecisionTree.get_count_by_attribute_value(parent_rows, self.target_attribute)
            return DecisionTree.DecisionNode(label=DecisionTree.get_max_value_in_dictionary(label_map))

        best_gain = -np.inf
        best_criteria = None
        best_attribute_partitions = None

        # Find the attribute having the best split "

        best_attribute_partitions, best_criteria = self.get_best_attribute_for_split(attribute_list,
                                                                                     best_attribute_partitions,
                                                                                     best_criteria, best_gain,
                                                                                     rows, splitting_func,
                                                                                     value_before_split)
        branches = {}
        for domain_value in self.attribute_domains[best_criteria]:
            branch_attr_list = list(attribute_list)
            branch_attr_list.remove(best_criteria)
            if domain_value in best_attribute_partitions.keys():
                partition_dataset = best_attribute_partitions[domain_value]
                branches[domain_value] = self.build_tree(rows=partition_dataset,
                                                         attribute_list=branch_attr_list,
                                                         parent_rows=rows,
                                                         depth=depth+1)
            else:
                branches[domain_value] = self.build_tree(rows=[],
                                                         attribute_list=branch_attr_list,
                                                         parent_rows=rows,
                                                         depth=depth+1)
        return DecisionTree.DecisionNode(attribute_name=best_criteria, branches=branches)

    def get_best_attribute_for_split(self, attribute_list, best_attribute_partitions,
                                     best_criteria, best_gain, rows,
                                     splitting_func, value_before_split):
        for attribute in attribute_list:
            if attribute == self.target_attribute:
                continue
            partitions = DecisionTree.perform_partition(rows, attribute)
            value_after_split = 0.0
            for partition_dataset in partitions.values():
                p = float(len(partition_dataset)) / len(rows)
                value_after_split += (p * splitting_func(partition_dataset))
            gain = value_before_split - value_after_split
            if gain > best_gain:
                best_gain = gain
                best_attribute_partitions = partitions
                best_criteria = attribute
        return best_attribute_partitions, best_criteria

    def print_decision_tree(self, decision_tree, indent="", default_indent=""):
        r"""
        Method to pretty print a decision tree
        to clearly indicate the decision criterias
        :param decision_tree: the current node of tree
        :param indent: indentation parameter
        :param default_indent: delta in indentation at ever branch
        :return:
        """
        # Is this a leaf node?
        if decision_tree.label is not None:
            print()
            print(indent + self.target_attribute + " : " + str(decision_tree.label))
            print(indent + "-" * 5 + "Leaf" + "-"*5)
            print()
        else:
            # Print the criteria
            print()
            print(indent + "Check " + str(decision_tree.attribute_name) + " ?")
            print(indent + "="*15)
            # Print the branches
            for partition_key, partition_tree in decision_tree.branches.items():
                print(indent + "If " + str(decision_tree.attribute_name) + " is " + partition_key + ":", )
                self.print_decision_tree(decision_tree=partition_tree,
                                         indent=indent+default_indent,
                                         default_indent=default_indent)

    @staticmethod
    def predict_row(decision_node, row):
        r"""
        Function to predict the label for the row according
        to the decision tree provided.
        :param decision_node: current node point
        :param row: the row for which prediction needs to be made
        :return: predicted label
        """
        if decision_node.label is not None:
            return decision_node.label
        else:
            target_value = row[decision_node.attribute_name]
            return DecisionTree.predict_row(decision_node=decision_node.branches[target_value], row=row)

    def predict(self):
        r"""
        Function to perform the testing of the learnt decision tree
        :return:
        """
        self.get_test_data()
        predicted_labels = []
        for row in self.test_data:
            predicted_labels.append(DecisionTree.predict_row(self.classifier, row))
        return predicted_labels

    @staticmethod
    def find_depth_tree(root):
        r"""
         Returns the depth of tree
        :param root: the root of tree
        :return:
        """
        if root is not None:
            max_depth = 0
            if root.branches is None:
                return 1
            else:
                for value in root.branches.values():
                    max_depth = max(max_depth, DecisionTree.find_depth_tree(value))
                return 1 + max_depth
        else:
            return 1

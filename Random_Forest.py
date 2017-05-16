from random import seed
from random import randrange
from csv import reader
from math import sqrt
import pandas as pd
import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from datetime import datetime
from sklearn import preprocessing
from scipy.cluster.vq import kmeans, vq
import bokeh.plotting
from bokeh.plotting import figure



from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, MinMaxScaler

# Random Sampling
def random_sampling(data_frame, fraction):
    data = data_frame.sample(n=(int)(len(data_frame) * fraction))
    return data


# function createFile
# To write the stratified sampled dataframe to csv
# Attributes
# random_data- the dataframe which has to be written
# file_name- name of the csv file
#
# Return -None
# Writes the dataframe to specifiedcsv

def createFile(random_data, file_name):
    random_data.to_csv(file_name, index=False, header=False)


# function createClusters
# To randomly sample the clusters formed by using kmeans
# Attributes
# dataFrame- the dataframe from which the dictionary has to be formed
#
# Return -
# A dictionary where key represents cluster number and value represents the indices of the
# rows in original dataframe

def createClusters(dataFrame):
    model = KMeans(n_clusters=5)
    model.fit(dataFrame)
    clusassign = model.predict(dataFrame)
    lables = model.labels_
    return lables


# function groupClusters
# To group the clusters to form a dictionary
# Attributes
# dataFrame- the dataframe from which the dictionary has to be formed
#
# Return -
# A dictionary where key represents cluster number and value represents the indices of the
# rows in original dataframe

def groupClusters(dataFrame):
    my_dict = {}
    for (ind, elem) in enumerate(createClusters(dataFrame)):
        if elem in my_dict:
            my_dict[elem].append(ind)
        else:
            my_dict.update({elem: [ind]})
    return my_dict


# function sampleClusters
# To randomly sample the clusters formed by using kmeans
# Attributes
# dataFrame- the dataframe which has to be sampled
#
# Return -
# A randomly sampled dataframe

def sampleClusters(dataFrame):
    cluster_sample = {}
    df = pd.DataFrame()

    y = dataFrame.iloc[:, -1]
    # print("last column:",dataFrame.columns[-1]);


    X = dataFrame.ix[:, dataFrame.columns != dataFrame.columns[-1]]

    colNo = []

    for i in range(len(X.columns)):
        colNo.append(i);

    # scaler = MinMaxScaler()
    # X_std = pd.DataFrame(scaler.fit_transform(dataFrame),columns= dataFrame.columns )
    #X_std = preprocessing.StandardScaler().fit(X[colNo]).transform(X[colNo])
    X_std = preprocessing.scale(X)

    findK(X_std)

    #showClusters(X_std)

    for i in range(0, 5):
        cluster_sample[i] = random_sample(groupClusters(dataFrame)[i])
        for k in cluster_sample[i]:
            df = df.append(dataFrame.iloc[[k]], ignore_index=True)
    return df

# def showClusters(dataset):
#     bokeh.plotting.output_notebook()
#     model = KMeans(n_clusters = 5)
#     model.fit(dataset)
#     i = 0
#
#     for sample in dataset:
#         if model.labels_[i] == 0:
#             plt.circle(x=sample[0], y=sample[1], size=15, color="red")
#         if model.labels_[i] == 1:
#             plt.circle(x=sample[0], y=sample[1], size=15, color="blue")
#
#         if model.labels_[i] == 2:
#             plt.circle(x=sample[0], y=sample[1], size=15, color="purple")
#
#         i + 1

# function findK
# To find the optimum number of clusters to be formed
# Attributes
# dataset- the dataframe which has to be analyzed
#

# Return - none
# 	Plots an elbow plot which helps in identification of cluster size

def findK(dataset):
    kValues = range(1, 8)
    meandist = []

    for k in kValues:
        model = KMeans(n_clusters=k)
        model.fit(dataset)
        clusassign = model.predict(dataset)
        meandist.append(sum(np.min(cdist(dataset, model.cluster_centers_, 'euclidean'), axis=1))
                        / dataset.shape[0])

    plt.plot(kValues, meandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')
    plt.show()


def random_sample(dataset):
    return random.sample(dataset, int(len(dataset) * 0.4))


# function load_csv
# To load the csv file as a dataset
# Attributes
# filename- name of the file to be loaded
#
# Return
# dataset as a list

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# function conv_String_to_float
# To convert the values of all the columns except the last to float from string
# Attributes
# dataset-the data read from the csv file
# column- the column that has to be converted
#
# Return
# dataset as a list

# def conv_String_to_float(dataset, column):
# 	for row in dataset:
# 		row[column] = float(row[column].strip())

# function conv_str_to_int
# To convert the values of last column which represents the class number from string
# to integers
# Attributes
# dataset-the data read from the csv file
# column- the column that has to be converted
#
# Return-
# dictionary- returns a
#
def conv_str_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    dictionary = dict()
    for i, value in enumerate(unique):
        dictionary[value] = i
    for row in dataset:
        row[column] = dictionary[row[column]]
    return dictionary


# function conv_str_to_int
# To convert the values of last column which represents the class number from string
# to integers
# Attributes
# dataset-the data read from the csv file
# column- the column that has to be converted
#
# Return-
# dictionary- returns a
#
def cross_validation_split(dataset, folds_count):
    split_dataFrame = list()

    dataset_copy = list(dataset)

    fold_length = int(len(dataset) / folds_count)

    for i in range(folds_count):
        group_data = list()
        while len(group_data) < fold_length:
            index = randrange(len(dataset_copy))
            group_data.append(dataset_copy.pop(index))
        split_dataFrame.append(group_data)
    return split_dataFrame

# function calc_accuracy
# To calculatethe accuracy of the algorithm
#
# Attributes
# actualValue-The actual outcome which identifies the class
# predictedValue- the value predicted by the decision tree algorithm
#
# Return-
# Accuracy percentage of the algorithm
#

def calc_accuracy(actualValue, predictedValue):
    correct_Predictions = 0

    for i in range(len(actualValue)):
        if actualValue[i] == predictedValue[i]:
            correct_Predictions += 1

    return correct_Predictions / float(len(actualValue)) * 100.0


def evaluate_algorithm(dataset, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = random_forest(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = calc_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def calc_gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 10000, 10000, 10000, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = calc_gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)

def find_PCA(dataframe):
    pca = PCA(n_components=2)
    pca.fit(dataframe)
    pca = pca.transform(dataframe)
    new_df = pd.DataFrame(pca)
    new_df.columns = ['A', 'B']
    plt.scatter(new_df.A, new_df.B, c=['r', 'b'])
    plt.show()

def screePlot(dataframe):
    pca = PCA(n_components=57)
    pca.fit(dataframe)
    y = pca.explained_variance_
    x = np.arange(len(y))+1
    plt.plot(x,y)
    plt.show()

seed(datetime.now())

df = pd.read_csv("Dataset.csv", header=None)

# random_sample = random_sampling(df, 0.2)
stratified_df = sampleClusters(df)
temp = stratified_df.ix[:, stratified_df.columns != stratified_df.columns[-1]]

find_PCA(temp)
screePlot(temp)

createFile(stratified_df, "data_Stratified.csv")
filename = 'data_Stratified.csv'
dataset = load_csv(filename)

conv_str_to_int(dataset, len(dataset[0]) - 1)

n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0]) - 1))
for n_trees in [2, 4]:
    scores = evaluate_algorithm(dataset, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
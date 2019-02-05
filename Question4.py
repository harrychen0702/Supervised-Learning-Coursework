import numpy as np
import matplotlib.pyplot as pl
import scipy.io as scio
import random
import Question1

# 14 variables
# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's


def split_data_set(data):     # 0<=index<20
    # we get 506 entries
    count = 0
    testing_set_index = []
    while count < 168:
        temp_index = random.randint(0, 505)
        if temp_index not in testing_set_index:
            testing_set_index.append(temp_index)
            count += 1
    testingSet = []
    for index in testing_set_index:
        testingSet.append(data[index])
    # print len(testing_set_index)
    trainingSet = []
    i = 0
    while i < 506:
        if i not in testing_set_index:
            trainingSet.append(data[i])
        i += 1
    new_data = []
    new_data.append(trainingSet)
    new_data.append(testingSet)
    return new_data     # return an array, with first element training set and second element testing set


def calculate_constant_MSE(trainging_y, testing_y, constant_y):
    training_sse = 0
    testing_sse = 0
    for y in trainging_y:
        training_sse += (y-constant_y) **2

    for y in testing_y:
        testing_sse += (y-constant_y) **2
    result = []
    result.append(training_sse/len(trainging_y))
    result.append(testing_sse/len(testing_y))
    # print result
    return result


#  Question (a)
def baseline_regression(trainingSet, testingSet):
    training_y = []
    testing_y = []
    i = 0
    while i < 338:
        training_y.append(trainingSet[i][13])
        i += 1
    i = 0
    while i < 168:
        testing_y.append(testingSet[i][13])
        i += 1
    y_average = sum(training_y) / len(training_y)
    result = calculate_constant_MSE(training_y, testing_y, y_average)
    return result


# Question (c)
def calculate_MSE(w, x, y):
    mse = ((x.dot(w) - y).T.dot((x.dot(w) - y))) / (len(y))
    return mse


# Linear regression with single attribute, plus bias term, return the mse of the prediction based on this attribute
def linear_regression_with_single_attribute(training_set, testing_set, k):  # k is the index of that attribute in data array, 0<=k<13
    x = np.ones((len(training_set), 2))  # 2d matrix with 2 columns
    array_y = []
    row = 0
    for d in training_set:
        x[row][0] = d[k]
        array_y.append(d[13])
        row += 1
    y = np.array(array_y)
    w = np.around(Question1.calculate_regression_coefficient(x, y), 5)
    training_mse = calculate_MSE(w, x,y)

    # Using the testing set to calculate mse
    test_x = np.ones((len(testing_set), 2))
    test_array_y = []
    row = 0
    for t in testing_set:
        test_x[row][0] = t[k]
        test_array_y.append(t[13])
        row += 1
    test_y = np.array(test_array_y)
    testing_mse = calculate_MSE(w, test_x, test_y)
    result = []
    result.append(np.around(training_mse,5))
    result.append(np.around(testing_mse, 5))
    return result


# Linear regression with all attributes, plus a bias term, so w has 14 elements, xi has 14 elements as well, with last one= 1
def linear_regression_with_all_attributes(training_set, testing_set):
    x = np.ones((len(training_set), 14))  # 2d matrix with 14 columns
    array_y = []
    row = 0
    for t in training_set:
        # put the elements of t into the row
        column = 0
        while column < 13:
            x[row][column] = t[column]
            column += 1
        array_y.append(t[13])
        row += 1
    y = np.array(array_y)
    w = np.around(Question1.calculate_regression_coefficient(x, y), 7)
    # print w

    error = []

    # calculate mse on training set
    training_error = calculate_MSE(w, x, y)

    # calculate mse on testing set
    test_x = np.ones((len(testing_set), 14))
    test_array_y = []
    row = 0
    for t in testing_set:
        column = 0
        while column < 13:
            test_x[row][column] = t[column]
            column += 1
        test_array_y.append(t[13])
        row += 1
    test_y = np.array(test_array_y)
    testing_error = calculate_MSE(w, test_x, test_y)

    error.append(training_error)
    error.append(testing_error)
    return error


if __name__ == "__main__":
    data_File = "boston.mat"
    data = scio.loadmat(data_File)['boston'] # data contains 506 entries, each entry is an array of 14 attributes
    # baseline_regression(data)

    # Question 4 part 1, generate the coefficients of regression function of each attribte
    # k = 0
    # while k<13:
    #     linear_regression_with_single_attribute(data, k)
    #     k += 1


    # Question 4 (c), print average mse on testing set for each attribute
    # k = 0
    # ks = np.arange(1, 14, 1)
    # result = []  # the average mse in for each attribute after 20 runs
    # while k < 13:    # Test each attribute
    #     loop = 0    # For each attribute, loop it 20 times to get average mse
    #     sum_mse = 0
    #     while loop < 20:
    #         mse = linear_regression_with_single_attribute(data, k)
    #         sum_mse += mse
    #         loop += 1
    #     result.append(np.around(sum_mse/20, 5))
    #     k += 1
    # print result
    # pl.plot(ks, np.array(result), '*')
    # pl.show()


    # Question 4 (d)
    result = [0, 0]  # mse on training set, mse on testing set
    loop = 0
    while loop < 20:
        temp = linear_regression_with_all_attributes(data)
        result[0] += temp[0]
        result[1] += temp[1]
        loop += 1
    result[0] = result[0]/20
    result[1] = result[1]/20
    print result

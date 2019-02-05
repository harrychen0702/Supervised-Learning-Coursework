import numpy as np
import scipy.io as scio
import random
import Question4
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

def split_training_data_set(data):     # 0<=index<20
    # split the data into 5 subsets, 335 entries in total
    random.shuffle(data)  # make the data set in random order
    result = []
    result.append(data[0:67])
    result.append(data[67:134])
    result.append(data[134:201])
    result.append(data[201:268])
    result.append(data[268:335])
    return result


def create_vectors():
    gamma_set = []
    variance = []
    index = -40
    while index <= -26:
        gamma_set.append(2 ** index)
        index += 1
    index = 7
    while index <= 13:
        variance.append(2 ** index)
        index += 0.5
    result = []
    result.append(gamma_set)
    result.append(variance)
    return result


def gaussian_kernel(xi, xj, var): # xi and xj are two nd-array, with size 13*1
    xs = xi - xj
    norm = 0
    for x in xs:
        norm += x ** 2
    result = np.exp(-1 * norm / (var ** 2) / 2)
    return result


def generate_k_matrix(trainingSet_x, var):  # Return a 2d nd-array.
    k_matrix = np.zeros((len(trainingSet_x), len(trainingSet_x)))  # Declare a 2d matrix with size 268*268
    i = 0
    while i < len(trainingSet_x):
        j = 0
        while j < len(trainingSet_x):
            k_matrix[i][j] = gaussian_kernel(trainingSet_x[i], trainingSet_x[j], var)
            j += 1
        i += 1
    return k_matrix


def calculate_a_star(k_matrix, gamma, l, trainingSet_y):    # l is 268
    I = np.zeros((len(trainingSet_y), len(trainingSet_y)))
    index = 0
    while index < len(trainingSet_y):
        I[index][index] = 1
        index += 1
    row = 0
    while row < I.shape[0]:
        column = 0
        while column <I.shape[1]:
            I[row][column] = I[row][column] * gamma * l
            column += 1
        row += 1
    temp = (k_matrix + I)
    inverse = np.linalg.inv(temp)
    a_star = inverse.dot(trainingSet_y)
    return a_star  # matrix with shape 268 *1


def get_y_test(l, a_star, traingingSet_x, x_test, var):
    i = 0
    ytest = 0
    while i < l:  # l = 268
        ytest += a_star[i] * gaussian_kernel(traingingSet_x[i],x_test, var)
        i += 1
    return ytest


def five_fold_cross_validation(data, gamma_set, variance):
    # Get the training_set with 335 entries, split it into 5 subsets
    splitted_data = split_training_data_set(data)  # has 5 sub-arrays, each with 67 entries
    predictor = [-1, -1, -1]  # [gamma, var, mse]
    threeD_points=[[],[],[]]  # Array of arrays, variance, gamma, mse
    for gamma in gamma_set:
        for var in variance:    # do cross validation for this pair, and calculate a mse, store it
            i = 0
            five_mses = 0
            while i < 5:  # Do validation for 5 times
                testingSet = splitted_data[i]   # testingSet is an array of 67 entries
                traingingSet = []  # trainingSet is an array of 268 entries
                j = 0
                while j < 5:
                    if j != i:
                        for d in splitted_data[j]:
                            traingingSet.append(d)
                    j += 1
                i += 1
                traingingSetArray_y=[]
                for t in traingingSet:
                    traingingSetArray_y.append(t[13])
                traingingSetArray_x=[]
                for t in traingingSet:
                    traingingSetArray_x.append(t[0:13])
                traingingSet_x = np.array(traingingSetArray_x)
                traingingSet_y = np.array(traingingSetArray_y)

                k_matrix = generate_k_matrix(traingingSet_x, var)
                # a* of shape 268 *1
                a_star = calculate_a_star(k_matrix, gamma, len(traingingSet_x), traingingSet_y)
                # print "a* is : ", a_star

                testingSetArray_y = []   # Now use the testingSet to test a*
                for t in testingSet:
                    testingSetArray_y.append(t[13])
                testingSetArray_x = []
                for t in testingSet:
                    testingSetArray_x.append(t[0:13])
                testingSet_x = np.array(testingSetArray_x)  # 67 * 13
                testingSet_y = np.array(testingSetArray_y)  # 67 * 1

                # Calculate the mse in this folder
                l = len(a_star)
                index = 0
                sse = 0
                while index < len(testingSet_y):
                    y_test = get_y_test(l, a_star, traingingSet_x, testingSet_x[index], var)
                    # print "ytest is: ",y_test
                    # print " y is: ", testingSet_y[index]
                    sse += (testingSet_y[index] - y_test) ** 2
                    index += 1
                mse = sse / len(testingSet_y)
                five_mses += mse
            five_mses /= 5
            print "var is ",var, "gamma is ",gamma, "mse is ", five_mses
            threeD_points[0].append(var)
            threeD_points[1].append(gamma)
            threeD_points[2].append(five_mses)
            if predictor[2] == -1 or predictor[2] > five_mses:
                predictor[0] = gamma
                predictor[1] = var
                predictor[2] = five_mses
    print threeD_points[0]
    print threeD_points[1]
    print threeD_points[2]
    return predictor


def calculate_training_mse(xSet, ySet, best_var, best_gamma):   # xSet is a 2d nd-array, ySet is a nd-array
    k_matrix = generate_k_matrix(xSet, best_var)
    l = len(ySet)
    a_star = calculate_a_star(k_matrix, best_gamma, l, ySet)
    index = 0
    sse = 0
    while index < len(ySet):
        y_test = get_y_test(l, a_star, xSet, xSet[index], best_var)
        sse += (ySet[index] - y_test) ** 2
        index += 1
    mse = sse / l
    return mse


def calculate_testing_mse(training_x, training_y, best_var, best_gamma, xSet, ySet):
    k_matrix = generate_k_matrix(training_x, best_var)
    l = len(training_y)
    a_star = calculate_a_star(k_matrix, best_gamma, l, training_y)
    index = 0
    sse = 0
    while index < len(ySet):
        y_test = get_y_test(l, a_star, training_x, xSet[index], best_var)
        sse += (ySet[index] - y_test) ** 2
        index += 1
    mse = sse / len(ySet)
    return mse


if __name__ == "__main__":



    # Question 5 (b)
    # data = create_vectors()
    # gamma_set = data[0]  # contains 15 numbers
    # variance = data[1]  # contains 13 numbers
    # data_File = "boston.mat"
    # data1 = scio.loadmat(data_File)['boston']  # data contains 506 entries, each entry is an array of 14 attributes
    # splitted_data = Question4.split_data_set(data1)
    # # best_predictor = five_fold_cross_validation(splitted_data[0], gamma_set, variance)
    # variance_points = threeD_points[0]
    # print variance_points
    # gamma_points = threeD_points[1]
    # print gamma_points
    # mse_points = threeD_points[2]
    # print mse_points
    # fig = pl.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('$Variance$', fontsize=15)
    # ax.set_ylabel(r'$\gamma$', fontsize=15)
    # ax.set_zlabel('$Mse$', fontsize=15)
    # ax.plot_trisurf(variance_points, gamma_points, mse_points, cmap=pl.get_cmap('rainbow'))
    # pl.show()


    # Question 5 (c)
    # data = create_vectors()
    # gamma_set = data[0]  # contains 15 numbers
    # variance = data[1]  # contains 13 numbers
    #
    # data_File = "boston.mat"
    # data1 = scio.loadmat(data_File)['boston']  # data contains 506 entries, each entry is an array of 14 attributes
    # splitted_data = Question4.split_data_set(data1)
    # # best_predictor = five_fold_cross_validation(splitted_data[0], gamma_set, variance)
    # # print best_predictor # Use this predictor to test
    # best_gamma = 2 ** (-40)
    # best_var = 2 ** 12.5
    #
    #
    # #Calculate mse on trainingSet
    # traingingSet = splitted_data[0]
    # traingingSetArray_y = []
    # for x in traingingSet:
    #     traingingSetArray_y.append(x[13])
    # traingingSetArray_x = []
    # for x in traingingSet:
    #     traingingSetArray_x.append(x[0:13])
    # trainingSet_x = np.array(traingingSetArray_x)
    # traingingSet_y = np.array(traingingSetArray_y)
    # print "mse on training set with best predictor is: "
    # print calculate_training_mse(trainingSet_x, traingingSet_y, best_var, best_gamma)
    #
    #
    # # Calculate mse on testingSet
    # testingSet = splitted_data[1]
    # testingSetArray_y = []
    # for t in testingSet:
    #     testingSetArray_y.append(t[13])
    # testingSetArray_x = []
    # for t in testingSet:
    #     testingSetArray_x.append(t[0:13])
    #
    # testingSet_x = np.array(testingSetArray_x)
    # testingSet_y = np.array(testingSetArray_y)
    # print "mse on testing set with best predictor is: ", calculate_testing_mse(trainingSet_x, traingingSet_y, best_var, best_gamma, testingSet_x, testingSet_y)


    # Question 5 (d)
    data = create_vectors()
    gamma_set = data[0]  # contains 15 numbers
    variance = data[1]  # contains 13 numbers

    data_File = "boston.mat"
    complete_data = scio.loadmat(data_File)['boston']  # data contains 506 entries, each entry is an array of 14 attributes

    # Do while loop for 20 times, each time split the complete data into trainingSet and testingSet, do mse on each model, add it to sse
    # At the end of 20 loops, do division on sse to get mse on each model
    loop = 0
    mse_training = np.zeros((16, 20))  # naive regression, single attribute regression from k=0 to k=12, all-attribute regression, kernel ridge regression
    mse_testing = np.zeros((16, 20))
    best_gamma = 2 ** (-40)
    best_var = 2 ** 12.5
    while loop < 20:
        splitted_data = Question4.split_data_set(complete_data)
        trainingSet = splitted_data[0]
        testingSet = splitted_data[1]
        index = 0  # index is the number of regression model, add on after calculating mse on on model
        mse1 = Question4.baseline_regression(trainingSet, testingSet)  # [mse_training, mse_testing]
        mse_training[index][loop] = mse1[0]
        mse_testing[index][loop] = mse1[1]
        index += 1
        while index < 14:
            mse_on_single = Question4.linear_regression_with_single_attribute(trainingSet, testingSet, index - 1)
            mse_training[index][loop] = mse_on_single[0]
            mse_testing[index][loop] = mse_on_single[1]
            index += 1
        mse_on_all_atttributes = Question4.linear_regression_with_all_attributes(trainingSet, testingSet)
        mse_training[index][loop] = mse_on_all_atttributes[0]
        mse_testing[index][loop] = mse_on_all_atttributes[1]
        index += 1

        # Calculate mse of kernel ridge regression on trainingSet
        trainingSetArray_y = []
        for x in trainingSet:
            trainingSetArray_y.append(x[13])
        trainingSetArray_x = []
        for x in trainingSet:
            trainingSetArray_x.append(x[0:13])
        trainingSet_x = np.array(trainingSetArray_x)
        trainingSet_y = np.array(trainingSetArray_y)
        mse_training[index][loop] = calculate_training_mse(trainingSet_x, trainingSet_y, best_var, best_gamma)

        # Calculate mse of kernel ridge regression on testingSet
        testingSetArray_y = []
        for t in testingSet:
            testingSetArray_y.append(t[13])
        testingSetArray_x = []
        for t in testingSet:
            testingSetArray_x.append(t[0:13])
        testingSet_x = np.array(testingSetArray_x)
        testingSet_y = np.array(testingSetArray_y)
        mse_testing[index][loop] = calculate_testing_mse(trainingSet_x,trainingSet_y, best_var, best_gamma, testingSet_x, testingSet_y)
        loop += 1


    # print mse_training
    # print mse_testing

    k = 0
    while k < 16:
        # calculate training mse
        training_mse = np.sum(mse_training[k]) / 20
        # calculate training mse standard deviation
        training_sd = np.std(mse_training[k])
        # calculate testing mse
        testing_mse = np.sum(mse_testing[k]) /20
        # calculate testing mse standard deviation
        testing_sd = np.std(mse_testing[k])
        print "k = ", k, " MSE train: ", training_mse, " sd: ", training_sd, " MSE test: ",testing_mse, " sd: ",testing_sd
        k += 1









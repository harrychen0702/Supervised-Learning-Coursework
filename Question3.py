import numpy as np
import matplotlib.pyplot as pl
import Question1

# Repeat 2 (b) with new basis


def map_data_matrix(data_x, k):    # New feature map, with basis sin(pi*x), sin(2*pi*x)........
    result = np.zeros((len(data_x), k))  # Declare a 2d matrix to store the data after feature map
    row = 0
    for x in data_x:
        column = 0
        while column < k:
            result[row][column] = (np.sin((column+1) * np.pi * x))
            column += 1
        row += 1
    return result


def calculate_MSE():
    ks = np.arange(1, 19, 1)      # Create dimensions array from 1 to 18
    x = np.random.uniform(0, 1, 30)     # Sample Xi uniformly at random
    y = []  # Array of y coordinates
    for data in x:  # Initialization of y coordinates
        random = np.random.normal(0, 0.07)
        temp = random + (np.sin(2 * np.pi * data)) ** 2
        y.append(temp)
    arr_x = np.array(x)
    arr_y = np.array(y)
    error = []
    for k in ks:
        mapped_data = np.around(map_data_matrix(arr_x, k), 7)
        print mapped_data
        coefficients = Question1.calculate_regression_coefficient(mapped_data, arr_y)
        coe = []
        for re in np.nditer(coefficients):
            coe.append(np.around(re, 7))
        # given k, get coefficients, for each of data in x, get estimated y, compare to original y, Calculate MSE
        new_coe = np.array(coe)
        result = (((mapped_data.dot(new_coe) - arr_y).T).dot((mapped_data.dot(new_coe) - arr_y))) / 30
        error.append(np.log(np.around(result,7)))
    pl.plot(ks, error, '*')
    pl.show()


# Repeat 2 (c) with new basis
def over_fitting():
    ks = np.arange(1, 19, 1)
    x = np.random.uniform(0, 1, 30)  # Sample Xi uniformly at random
    y = []  # Array of y coordinates
    mse_error = []

    # Generate testing set
    testing_x = np.random.uniform(0, 1, 1000)  # This is the test set, Sample Xi uniformly at random
    testing_y = []  # Array of y coordinates
    for data in testing_x:  # Initialization of y coordinates
        random = np.random.normal(0, 0.07)
        temp = random + (np.sin(2 * np.pi * data)) ** 2
        testing_y.append(temp)
    arr_testing_x = np.array(testing_x)
    arr_testing_y = np.array(testing_y)

    # Generating training set
    for data in x:  # Initialization of y coordinates
        random = np.random.normal(0, 0.07)
        temp = random + (np.sin(2 * np.pi * data)) ** 2
        y.append(temp)
    arr_x = np.array(x)
    arr_y = np.array(y)
    error = []
    for k in ks:
        mapped_data = np.around(map_data_matrix(arr_x, k), 5)
        print mapped_data
        coefficients = Question1.calculate_regression_coefficient(mapped_data, arr_y)
        new_coe = []
        for re in np.nditer(coefficients):
            new_coe.append(np.around(re, 7))
        # given k, get coefficients, for each of data in x, get estimated y, compare to original y, Calculate MSE
        training_mapped_data = np.around(map_data_matrix(arr_testing_x, k), 7)  # Apply feature map
        result = (((training_mapped_data.dot(new_coe) - arr_testing_y).T).dot(
            (training_mapped_data.dot(new_coe) - arr_testing_y))) / 1000
        error.append(np.log(result))
        mse_error.append(result)
    pl.plot(ks, error, '*')
    pl.show()
    return mse_error  # This is for question d (ii) only


# Repeat Question 2 (d) (i)
def average_100_MSE():
    # Run the for loop for 100 times, create an array of capacity 18,
    result_array = []   # Store the sum of mse for each k in each run, size of 18
    i = 0
    while i < 18:
        result_array.append(0)
        i += 1
    ks = np.arange(1, 19, 1)  # Create dimensions array from 1 to 18
    loop = 0
    while loop < 100:
        x = np.random.uniform(0, 1, 30)  # Sample Xi uniformly at random
        y = []  # Array of y coordinates
        for data in x:  # Initialization of y coordinates
            random = np.random.normal(0, 0.07)
            temp = random + (np.sin(2 * np.pi * data)) ** 2
            y.append(temp)
        arr_x = np.array(x)
        arr_y = np.array(y)
        error = []
        for k in ks:
            mapped_data = np.around(map_data_matrix(arr_x, k), 7)  # Apply feature map
            coefficients = np.around(Question1.calculate_regression_coefficient(mapped_data, arr_y), 7)
            # given k, get coefficients, for each of data in x, get estimated y, compare to original y, Calculate MSE
            result = ((mapped_data.dot(coefficients) - arr_y).T.dot((mapped_data.dot(coefficients) - arr_y))) / 30
            error.append(np.around(result, 7))
        index = 0
        while index < len(error):
            result_array[index] += error[index]
            index += 1
        loop += 1
    # While loop finished
    mse = []
    for re in result_array:
        mse.append(np.log(re/100))
    pl.plot(ks, mse, '*')
    pl.show()


# Repeat Question 2 (d) (ii)
def average_100_overfitting():
    ks = np.arange(1, 19, 1)  # Create dimensions array from 1 to 18
    result = []
    i = 0
    while i < 18:
        result.append(0)
        i += 1
    loop = 0
    while loop < 100:
        error_in_loop = over_fitting()  # Invoke the over_fitting function above
        loop += 1
        index = 0
        while index < 18:
            result[index] += error_in_loop[index]
            index += 1
    log_error=[]
    for re in result:
        log_error.append(np.log(re/100))
    pl.plot(ks, log_error, '*')
    pl.show()



if __name__ == "__main__":
    # calculate_MSE()
    # over_fitting()
    # average_100_MSE()
    average_100_overfitting()
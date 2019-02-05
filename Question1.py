import numpy as np
import math
import matplotlib.pyplot as pl


def map_data_matrix(data_x, k):    # Given k and x coordinates, apply feature map
    result = np.zeros((len(data_x), k))  # Declare a 2d matrix to store the data after feature map
    row = 0
    for x in data_x:
        column = 0
        while column < k:
            result[row][column] = math.pow(x, column)
            column += 1
        row += 1
    return result


def calculate_regression_coefficient(mapped_data, data_y):  # Calculate Regression function, return weights
    transpose = mapped_data.T
    temp = transpose.dot(mapped_data)  # Matrix cross product
    inverse = np.linalg.inv(temp)
    result = inverse.dot(transpose).dot(data_y)
    new_coe = []
    for re in np.nditer(result):
        new_coe.append(np.around(re, 2))    # We must keep up to 2 decimal places to avoid wrong answer
    #return np.array(new_coe)   # result is nd-array
    return result


def draw_polynomial(arr_x, arr_y):
    k = 1
    pl.figure(figsize=(8, 4))
    fig, ax = pl.subplots()  # To superimposing four graphs together
    while k <= 4:
        mapped_data = map_data_matrix(arr_x, k)  # Apply feature map
        k += 1
        coefficients = calculate_regression_coefficient(mapped_data, arr_y)
        print coefficients
        print (mapped_data.dot(coefficients) - arr_y).T.dot((mapped_data.dot(coefficients) - arr_y)) / 4
        x = np.linspace(0, 5, 100)
        y = 0
        index = 0
        for coefficient in np.nditer(coefficients):
            y = y + coefficient * x**index  # Generate polynomial functions
            index += 1
        ax.plot(x, y, color="blue", linewidth=1.5)
    pl.show()


if __name__ == '__main__':
    data_x = [1, 2, 3, 4]
    data_y = [3, 2, 0, 5]

    arr_x = np.array(data_x)  # Convert the python list to ndarray
    arr_y = np.array(data_y)

    mapped_data = map_data_matrix(arr_x, 3)  # Apply feature map
    #coefficients = calculate_regression_coefficient(mapped_data, arr_y)
    draw_polynomial(arr_x, arr_y)



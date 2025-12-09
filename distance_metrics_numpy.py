import numpy as np

# manhattan distance(L1)
def manhattan_distance_L1(A, B):
    difference = A - B
    absolute_difference = np.abs(difference)
    L1_distance = np.sum(absolute_difference)
    return L1_distance

# euclid distance(L2)
def euclid_distance_L2(A, B):
    difference = A - B
    squared_difference = difference**2
    sum_of_squares = np.sum(squared_difference)
    L2_distance = np.sqrt(sum_of_squares)
    return L2_distance
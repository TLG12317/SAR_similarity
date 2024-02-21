#Find all the permutations of an nxn matrix that is translation invariant
#O(n**4) algorithm
import itertools
import numpy as np

import cv2

def permutations(n):
    #Usage of placeholder matrix and turning it into a 2x2 numpy array
    all_permutations = set()
    lst_of_mat = list(itertools.product([0, 1], repeat=n**2))
    unique = set()

    for matrix in lst_of_mat:
        matrix = np.array(matrix)
        matrix = np.reshape(matrix, (n,n))
        original_matrix = np.copy(matrix)
        original_matrix = np.reshape(original_matrix, (1,n**2))
        #All translational invariant permutations for given nxn matrix
        for dr in range(n):
            matrix = np.roll(matrix, 1, axis=0) # shift 1 place in horizontal axis
            for dc in range(n):
                matrix = np.roll(matrix, 1, axis=1) # shift 1 place in vertical axis
                to_store = np.reshape(matrix, (1,n**2))
                if tuple(to_store[0]) in all_permutations:
                    continue
                else:
                    unique.add(tuple(original_matrix[0]))
                    all_permutations.add(tuple(to_store[0])) #store in dictionary


    lst = list(unique)


    for i in range(len(lst)):
        lst[i] = list(lst[i])
        lst[i] = np.array(list(lst[i]))
        lst[i] = lst[i].astype(np.uint8)
        lst[i]  = np.reshape(lst[i], (n,n))

    return lst

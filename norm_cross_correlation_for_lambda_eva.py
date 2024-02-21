import itertools
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.linalg.lapack import dpstrf

from permutations import permutations
from Optimizing_norm_x import Optimizing_norm_x
from get_NCC_with_rank_constraints import get_NCC_with_rank_constraints



def norm_cross_correlation(lst, dimensions):
    dct_of_mat = {}
    lst_of_mat = []
    for i in range(len(lst)):
        lst[i] = np.array(list(lst[i]))
        lst[i] = lst[i].reshape(dimensions,dimensions)
        lst[i] = lst[i].astype(np.float32)
        lst_of_mat.append(f"mat{i+1}")
        dct_of_mat[f"mat{i+1}"] = lst[i] 

    x = set()
    ret = []
    for i in range(len(lst)):
        lst2 = []
        mat = lst[i]
        mat = np.pad(mat, max(len(mat), len(mat[0])), mode='wrap') #padded lst[i]
        for j in range(len(lst)):
            res = cv2.matchTemplate(mat,lst[j], cv2.TM_CCORR_NORMED)
            value = cv2.minMaxLoc(res)[1]
            lst2.append(value)
        ret.append(lst2.copy())
    return ret

def Getting_Vector_Representation_Matrix(matrix_lst, rank_constraint, dimensions):
    """
    :INPUTS:
    matrix_lst: List of nxn dimensional matrices 
    rank_constraint: Final Rank of NCC matrix

    :OUTPUTS:
    A_prime: the vector representation matrix of matrix_lst
    
    Takes in a list of matrices and a rank_constraint, and outputs matrix A_prime, 
    that maps each matrix in the list to a vector
    """
    G_prime = Rank_Reduction_Positive_Semidefinite_G(matrix_lst, rank_constraint, dimensions)
    A_prime = Cholesky_Decomposition(G_prime, rank_constraint)

    return A_prime

def Getting_Vector_Estimation_of_Image():
    with open("b_under_50_50.txt") as f:
        b_under = f.readlines()


    vector_estimation_of_image = []
    for i in range(len(b_under)):
        temp = b_under[i][:-1].split(",")
        temp_new = []
        for j in range(len(temp)):
            temp_new.append(float(temp[j]))
        vector_estimation_of_image.append(np.array(temp_new))

    vector_estimation_of_image = np.array(vector_estimation_of_image)
    return vector_estimation_of_image

def Rank_Reduction_Positive_Semidefinite_G(matrix_lst, rank_constraint, dimensions):
    """
    :INPUTS:
    matrix_lst: List of nxn dimensional matrices 
    rank_constraint: Final Rank of NCC matrix

    :OUTPUTS:
    G_prime: Rank Reduced and positive semidefinite NCC Matrix 

    Takes in a list of matrices and a rank_constraint, and outputs the NCC matrix with said rank_constraint
    """
    G = norm_cross_correlation(matrix_lst, dimensions) 
    get_NCC_with_rank_constraints(G, rank_constraint)

    with open('output_X.txt') as f:
        new_G = f.readlines()

    G_prime = [] #rank determined by rank constraint
    for i in range(len(new_G)):
        temp = new_G[i][:-1].split(",")
        temp_new = []
        for j in range(len(temp)):
            temp_new.append(float(temp[j]))
        G_prime.append(np.array(temp_new))

    G_prime = np.array(G_prime) 
    return G_prime

def Cholesky_Decomposition(G, rank):
    """
    :INPUTS:
    G: The rank reduced positive semidefinite NCC matrix

    :OUTPUTS:
    A: The vector representation matrix 
    
    """
    G = 0.5*(G+G.T)
    G = G - np.diag(np.diag(G)) + np.identity(len(G)) * 1.000001 

    A = np.linalg.cholesky(G) #G = A*A.T
    G = G - np.diag(np.diag(G)) + np.identity(len(G)) 
    A[:, rank:] = 0

    return A

#remaining matrix to compare to the 50 and see what original value was

#solve x for Ax = b as it only uses 50 values 
#compare x to the values obtained in original when comparing to the 63 matrices

#this i sboringgggggggggg
#ahhhhhh gremlin gremlin gremlin gremlin
#my neck and bacjkhurts

def main(dimensions, constant_of_proportionality_enhancer, sample, rank_constraint):
    """
    After inputting the dimensions of the matrix that you want to look at (nxn matrix)
    and the sample subset of the number of matrices that you want to look at

    We denote G as the initial NCC table, which is a matrix of NCC values

    Firstly the code <Pencorr.m> will reduce the rank of G, as well as
    make G a positive semi definite (eigenvalues are non negative) 
    and Hermitian (symmetrical)
    matrix with diagonals that are equal to 1 (preserves unit norm)

    We denote G_prime as the NCC table of a selected subset of matrices, and run Pencorr.m
    A_prime is defined as the vector representation matrix of the subset of matrices
    The rank of A_prime will be equivalent to G_prime

    Using the formula A_prime_T.A_prime = G_prime,
    we use a Cholesky decomposition to find A

    We denote b as a matrix of dimension (len(subset) x 1)
    Let Q be a random matrix from G that was not selected
    b will contain the NCC values of the subset of selected matrices and Q
    Using A_prime(x1) = b, the code (SolveLsNormConst.m) will solve for x, whilst maintaining |x| = 1

    We denote x1 as the vector representation for the one random matrix from G that was not selected

    G is defined as the NCC table of all matrices
    A is defined as the vector representation matrix of all matrices

    We denote the vector embedding of Q in A as x2
    """
    
    #find all matrices of nxn dimensions
    test_name = f"dim_{dimensions}_sample_{sample}_proportional_{constant_of_proportionality_enhancer}"
    total_perm_of_matrices = permutations(dimensions)
    zeroes = np.array([0 for _ in range(dimensions**2)])
    zeroes = zeroes.reshape((dimensions,dimensions))
    for k in range(len(total_perm_of_matrices)):
        if (total_perm_of_matrices[k] == zeroes).all():
            total_perm_of_matrices.pop(k)
            break

    #sample a subset of the matrices
    numbers = random.sample(range(len(total_perm_of_matrices)), sample+1) #picking subset of matrices and 1 special matrix
    special_num = numbers[-1]
    numbers = numbers[:-1]

    #Denote special_mat as a random nxn matrix not in the subset
    special_mat = total_perm_of_matrices[special_num] #one of the matrices from the remaning 13 that was part of the original 63x63 but not chosen to be in 50 x 50
    special_mat = special_mat.astype(np.float32)

    random_matrices = []
    for num in numbers:
        total_perm_of_matrices[num] = total_perm_of_matrices[num].astype(np.float32)
        random_matrices.append(total_perm_of_matrices[num])
        
    #b is the ncc between special_mat and each of the matrices
    b = []
    for j in range(len(random_matrices)):
        res = cv2.matchTemplate(special_mat, random_matrices[j], cv2.TM_CCORR_NORMED)
        b.append(res[0][0])
    b = np.array(b)

    #Vector representation matrix of the original set
    original_A_prime = Getting_Vector_Representation_Matrix(total_perm_of_matrices, rank_constraint, dimensions)
    vector_estimation_of_image_original = original_A_prime[special_num][:rank_constraint]

    #Vector representation matrix of the subset
    A_prime = Getting_Vector_Representation_Matrix(random_matrices, rank_constraint, dimensions)

    #Optimises x in the equation A_prime(x) = b, x is the vector estimation of image
    Optimizing_norm_x(A_prime, b)
    vector_estimation_of_image = Getting_Vector_Estimation_of_Image()

    error = 0
    for i in range(sample):
        error += (vector_estimation_of_image[i] - vector_estimation_of_image_original[i])**2

    constant_of_proportionality = sample/len(total_perm_of_matrices) # ratio < 1

    for i in range(len(vector_estimation_of_image)): #manipulating the sample
        vector_estimation_of_image[i] *= (constant_of_proportionality)**constant_of_proportionality_enhancer

    final_lst = []
    for i in range(len(numbers)):
        final_lst.append((numbers[i], vector_estimation_of_image[i], vector_estimation_of_image_original[i]))
    final_lst.sort(key=lambda x:x[0])

    #plot graph of x being matrix number and y being the new values
    #then plot a graph of differences?

    fig, ax = plt.subplots()
    ax.plot([x[0] for x in final_lst], [x[1] for x in final_lst], label="line50x50")
    ax.plot([x[0] for x in final_lst], [x[2] for x in final_lst], label="line63x63")

    ax.set(xlabel='matrix number', ylabel='ncc_value',
        title='ncc_val comparison')
    ax.grid()
    fig.savefig(test_name + ".png")
    plt.show()

if __name__ == "__main__":
    dimensions = 3
    constant_of_proportionality_enhancer = 0
    sample = rank_constraint = 50 #make rank constraint equal to sample for now

    main(dimensions, constant_of_proportionality_enhancer, sample, rank_constraint)
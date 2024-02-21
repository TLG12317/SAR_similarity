import itertools
import numpy as np

import cv2

from permutations import permutations
from get_NCC_with_rank_constraints import get_NCC_with_rank_constraints


dimensions = 3
lst = permutations(dimensions)
# for i in range(len(lst)):
#     lst[i] = np.subtract(lst[i], np.mean(lst[i]))

def norm_cross_correlation(lst):
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

x = norm_cross_correlation(lst)
print(type(x))
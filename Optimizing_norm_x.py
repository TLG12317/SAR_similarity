import subprocess
import os

def Optimizing_norm_x(A, b):
    #Optimizes x in the formula Ax=b under the constraint that |x| == 1

    #A is the vector representation matrix of images
    #b is the NCC of one random image with the 50 images
    lst = []
    for i in range(len(A)):
        lst.append(list(A[i]))
    k = str(len(A))
    A = str(lst)
    b = str(b)

    A=A.replace("]", "\n")
    A=A.replace("[", "")
    A=A.replace(",", "")
    A = A[:-3]

    b=b.replace("]", "\n")
    b=b.replace("[", "")
    b=b.replace(",", "")
    b=b[:-1]

    with open(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp4.txt", "w") as f:
        f.writelines(A)

    with open(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp5.txt", "w") as f:
        f.writelines(b)

    with open(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp6.txt", "w") as f: #length of the sample size
        f.writelines(k)

    #call the function to optimize x
    print("\n Optimizing x")
    subprocess.call(['octave.exe', "SolveLsNormConst.m"])

    
    os.remove(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp4.txt")
    os.remove(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp5.txt")
    os.remove(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp6.txt")

    #output is in the form of b_under_50_50.txt where x is the estimated normed vector representation of the random image


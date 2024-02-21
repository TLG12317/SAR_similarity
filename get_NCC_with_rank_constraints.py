import subprocess
import os


def get_NCC_with_rank_constraints(lst_of_mat, rank_constraint=None):
    #create temp files of ncc_table and length of ncc_table 
    print("--------GETTING THE NCC TABLE--------")
    rank_constraint = str(len(lst_of_mat)) if rank_constraint is None else rank_constraint
    n = str(len(lst_of_mat))

    lst_of_mat = str(lst_of_mat)
    lst_of_mat=lst_of_mat.replace("]", "\n")
    lst_of_mat=lst_of_mat.replace("[", "")
    lst_of_mat=lst_of_mat.replace(",", "")
    lst_of_mat = lst_of_mat[:-3]


    with open(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp.txt", "w") as f:
        f.writelines(lst_of_mat)

    with open(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp2.txt", "w") as f:
        f.writelines(n)

    with open("temp3.txt", 'w') as f:
        f.writelines(str(rank_constraint))

    print("-------RUNNING OCTAVE--------")
    #run the ncc_table through octave, output is new X of lower rank written in text file output_X.txt
    subprocess.call(['octave.exe', "testPenCorr.m"])

    #cleanup temp files

    os.remove(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp.txt")
    os.remove(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp2.txt")
    os.remove(r"C:\Users\LChengZe\Desktop\SAR_SIM_2\temp3.txt")


    #output is in the form of output_X.txt and output_X2.txt (residue)
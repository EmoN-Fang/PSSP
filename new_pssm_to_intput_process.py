import numpy as np
import os


output_count = 0
output_x_path = "/home/emon/Data/blast/200/pataa_npy/"
output_y3_path = "/home/emon/Data/blast/200/new_second_three_npy/"
output_y8_path = "/home/emon/Data/blast/200/new_second_eight_npy/"
input_y3_path = "/home/emon/Data/blast/200/second_three/"
input_y8_path = "/home/emon/Data/blast/200/second_eight/"

def readFile(filepath):
    global output_count
    f = open(filepath, "r")

    lines = f.readlines()
    lines_count = len(lines)

    # print(lines_count)
    input_matrix = list()

    for n in range(3, lines_count - 6):
        line = lines[n][7:]
        line = ' '.join(line.split())
        result = line.split(' ')
        for i in range(20):
            result[i] = int(result[i])
        input_matrix.append(result[0:20])

    final_matrix = np.array(input_matrix)
    # print(final_matrix)
    # print(final_matrix.shape)
    # print(final_matrix[0])
    # print(final_matrix[153])
    tmp_path = output_x_path + str(output_count) + ".npy"
    np.save(tmp_path, final_matrix)
    # print(final_matrix)
    # print(tmp_path)
    f.close()

def Process_3_y(file):
    global output_count
    g = open(file, "r")

    lines = g.readlines()
    #lines_count = len(lines)
    # print(lines_count)
    input_matrix = list()

    sec_list = lines[1]
    #print(sec_list)
    length = len(sec_list)
    #print(length)
    for i in range(length):
        if sec_list[i] == "H":
            input_matrix.append(0)
        if sec_list[i] == "E":
            input_matrix.append(1)
        if sec_list[i] == "C":
            input_matrix.append(2)

    final_matrix = np.array(input_matrix)
    out_path = output_y3_path + str(output_count) + ".npy"
    np.save(out_path, final_matrix)
    g.close()

def Process_8_y(file):
    global output_count
    h = open(file, "r")

    lines = h.readlines()
    #lines_count = len(lines)
    # print(lines_count)
    input_matrix = list()

    sec_list = lines[1]
    #print(sec_list)
    length = len(sec_list)
    #print(length)
    for i in range(length):
        if sec_list[i] == "G":
            input_matrix.append(0)
        if sec_list[i] == "H":
            input_matrix.append(1)
        if sec_list[i] == "I":
            input_matrix.append(2)
        if sec_list[i] == "B":
            input_matrix.append(3)
        if sec_list[i] == "E":
            input_matrix.append(4)
        if sec_list[i] == "S":
            input_matrix.append(5)
        if sec_list[i] == "T":
            input_matrix.append(6)
        if sec_list[i] == "C":
            input_matrix.append(7)

    final_matrix = np.array(input_matrix)
    out_path = output_y8_path + str(output_count) + ".npy"
    np.save(out_path, final_matrix)
    h.close()



def Process_pssm(filepath):
    global output_count
    pathDir = os.listdir(filepath)
    for s in pathDir:
        newDir = os.path.join(filepath, s)
        # print(os.path.splitext(s)[0])
        if os.path.splitext(newDir)[1] == ".pssm":
            readFile(newDir)
            input_y3_file = input_y3_path + str(os.path.splitext(s)[0])+".txt"
            Process_3_y(input_y3_file)
            input_y8_file = input_y8_path + str(os.path.splitext(s)[0])+".txt"
            Process_8_y(input_y8_file)
            output_count += 1

Process_pssm("/home/emon/Data/blast/200/pataa")

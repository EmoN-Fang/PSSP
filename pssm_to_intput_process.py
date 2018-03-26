import numpy as np


f = open("swissprot1.pssm", "r")

lines = f.readlines()
lines_count = len(lines)

#print(lines_count)
input_matrix = list()


for n in range(3, lines_count-6):
    line = lines[n][7:91]
    line = ' '.join(line.split())
    result = line.split(' ')
    for i in range(20):
        result[i] = int(result[i])
    input_matrix.append(result)

final_matrix = np.array(input_matrix)
print(final_matrix)
print(final_matrix.shape)
print(final_matrix[0])
print(final_matrix[153])

np.save("swissprot1.npy", final_matrix)

#     for i in range (20):
#         tmp_matrix = list()
#         tmp_matrix.append(int(result[i]))
#     input_matrix.append(tmp_matrix)

# final_matirx = np.array(input_matrix)

# print(final_matirx)


# tmp_matrix = list()
# tmp_matrix.append(int(-2))
# tmp_matrix.append(int(3))
# input_matrix.append(tmp_matrix)
# input_matrix.append(tmp_matrix)
# b = np.array(input_matrix)
# print(input_matrix)
# print(b.shape)


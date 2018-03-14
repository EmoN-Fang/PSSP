# # -*-coding: utf-8 -*-
# f = open("username.txt","w")
# f.write("Lycoridiata\n")
# f.write("wulei\n")
# f.write("leilei\n")
# f.write("Xingyu\n")
import numpy as np

f = open("ss.txt", "r")
# print(f.read())
get = f.read()
result = get.split('>')
# "".join(result.split())

sequence = list()
second_three = list()
second_eight = list()

for i in range(1, len(result)):
    # "".join(result[i].split( ))
    # re.split(r'\s+', result[i])
    result[i] = result[i].replace('\n', '')
    if (i % 2 == 1):
        sequence.append(result[i][15:])
        # sequence[i] = result[i][15:]
        print(sequence[(i - 1) // 2])
        print("******", len(sequence), "******")
    else:
        second_three.append(result[i][13:])
        print(second_three[(i - 2) // 2])
        print("******", len(second_three), "******")
    # print(i)
    # print(result[i])
    # print("******")


for i in range(2, len(result), 2):
    # "".join(result[i].split( ))
    # re.split(r'\s+', result[i])
    result[i] = result[i].replace('\n', '')
    result[i] = result[i].replace('G', 'H')
    result[i] = result[i].replace('I', 'H')
    result[i] = result[i].replace('B', 'E')
    result[i] = result[i].replace('S', 'C')
    result[i] = result[i].replace('T', 'C')
    second_eight.append(result[i][13:])
    print(second_eight[(i - 2) // 2])
    print("******", len(second_eight), "******")


f.close()

np.save("sequence.npy", sequence)
np.save("second_three.npy", second_three)
np.save("second_eight.npy", second_eight)
# np.array(sequence)
# np.array(second_three)
# np.array(second_eight)

# print(len(sequence))
# print(sequence[0][4])


# import sys
# import os
# import shutil

# f = open("username.txt", "r")
# lines = f.readlines()
# for line in lines:
#     line = line.strip('\n')
#     shutil.copy(line, 'username')

# a = 'hello world'

# b = a.replace(' ', '')

# print(a)
# print(b)

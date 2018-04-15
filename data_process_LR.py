import numpy as np

input_x_path = "/home/emon/Data/blast/100/pataa_npy/"
input_y3_path = "/home/emon/Data/blast/100/second_three_npy/"
input_y8_path = "/home/emon/Data/blast/100/second_eight_npy/"
output_folder = "/home/emon/Data/blast/100/"

tmp_x_path = input_x_path + str(0) + ".npy"
tmp_x = np.load(tmp_x_path)
X_train = tmp_x.T
for train_count in range(1, 35000):
    tmp_path = input_x_path + str(train_count) + ".npy"
    tmp_x = np.load(tmp_path)
    tmp_x = tmp_x.T
    X_train = np.hstack((X_train, tmp_x))
out_X_train_path = output_folder + "X_train.npy"
np.save(out_X_train_path, X_train)

tmp_y3_path = input_y3_path + str(0) + ".npy"
tmp_y3 = np.load(tmp_y3_path)
Y3_train = tmp_y3.T
for train_count in range(1, 35000):
    tmp_path = input_y3_path + str(train_count) + ".npy"
    tmp_y3 = np.load(tmp_path)
    tmp_y3 = tmp_y3.T
    Y3_train = np.hstack((Y3_train, tmp_y3))
out_Y3_train_path = output_folder + "Y3_train.npy"
np.save(out_Y3_train_path, Y3_train)

tmp_y8_path = input_y8_path + str(0) + ".npy"
tmp_y8 = np.load(tmp_y8_path)
Y8_train = tmp_y8.T
for train_count in range(1, 35000):
    tmp_path = input_y8_path + str(train_count) + ".npy"
    tmp_y8 = np.load(tmp_path)
    tmp_y8 = tmp_y8.T
    Y8_train = np.hstack((Y8_train, tmp_y8))
out_Y8_train_path = output_folder + "Y8_train.npy"
np.save(out_Y8_train_path, Y8_train)


tmp_x_path = input_x_path + str(35000) + ".npy"
tmp_x = np.load(tmp_x_path)
X_test = tmp_x.T
for test_count in range(35001, 35633):
    tmp_path = input_x_path + str(test_count) + ".npy"
    tmp_x = np.load(tmp_path)
    tmp_x = tmp_x.T
    X_test = np.hstack((X_test, tmp_x))
out_X_test_path = output_folder + "X_test.npy"
np.save(out_X_test_path, X_test)

tmp_y3_path = input_y3_path + str(35000) + ".npy"
tmp_y3 = np.load(tmp_y3_path)
Y3_test = tmp_y3.T
for test_count in range(35001, 35633):
    tmp_path = input_y3_path + str(test_count) + ".npy"
    tmp_y3 = np.load(tmp_path)
    tmp_y3 = tmp_y3.T
    Y3_test = np.hstack((Y3_test, tmp_y3))
out_Y3_test_path = output_folder + "Y3_test.npy"
np.save(out_Y3_test_path, Y3_test)

tmp_y8_path = input_y8_path + str(35000) + ".npy"
tmp_y8 = np.load(tmp_y8_path)
Y8_test = tmp_y8.T
for test_count in range(35001, 35633):
    tmp_path = input_y8_path + str(test_count) + ".npy"
    tmp_y8 = np.load(tmp_path)
    tmp_y8 = tmp_y8.T
    Y8_test = np.hstack((Y8_test, tmp_y8))
out_Y8_test_path = output_folder + "Y8_test.npy"
np.save(out_Y8_test_path, Y8_test)

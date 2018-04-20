import numpy as np

input_x_path = "/home/emon/Data/blast/100/pataa_npy/"
input_y3_path = "/home/emon/Data/blast/100/second_three_npy/"
input_y8_path = "/home/emon/Data/blast/100/second_eight_npy/"
output_folder = "/home/emon/Data/blast/100/LR_model_new/"


def zero_pad(X):

    X_pad = np.pad(X, ((7, 7), (0, 0)), 'constant', constant_values=(0, 0))

    return X_pad


X_train = np.zeros((300, 1))
for train_count in range(0, 30000):
    tmp_path = input_x_path + str(train_count) + ".npy"
    tmp_x = np.load(tmp_path)
    num_x = tmp_x.shape[0]
    tmp_x_pad = zero_pad(tmp_x)
    for i in range(num_x):
        small_x = tmp_x_pad[i:i + 15, :]
        small_x_flat = small_x.flatten()
        small_x_new = small_x_flat.reshape(300, 1)
        X_train = np.hstack((X_train, small_x_new))
out_X_train_path = output_folder + "X_train.npy"
np.save(out_X_train_path, X_train)

Y3_train = np.zeros((3, 1))
for train_count in range(0, 30000):
    tmp_path = input_y3_path + str(train_count) + ".npy"
    tmp_y3 = np.load(tmp_path)
    tmp_y3 = tmp_y3.T
    Y3_train = np.hstack((Y3_train, tmp_y3))
#Y3_train = Y3_train.T
out_Y3_train_path = output_folder + "Y3_train.npy"
np.save(out_Y3_train_path, Y3_train)


Y8_train = np.zeros((8, 1))
for train_count in range(0, 30000):
    tmp_path = input_y8_path + str(train_count) + ".npy"
    tmp_y8 = np.load(tmp_path)
    tmp_y8 = tmp_y8.T
    Y8_train = np.hstack((Y8_train, tmp_y8))
#Y8_train = Y8_train.T
out_Y8_train_path = output_folder + "Y8_train.npy"
np.save(out_Y8_train_path, Y8_train)


X_test = np.zeros((300, 1))
for test_count in range(30000, 31000):
    tmp_path = input_x_path + str(test_count) + ".npy"
    tmp_x = np.load(tmp_path)
    num_x = tmp_x.shape[0]
    tmp_x_pad = zero_pad(tmp_x)
    for i in range(num_x):
        small_x = tmp_x_pad[i:i + 15, :]
        small_x_flat = small_x.flatten()
        small_x_new = small_x_flat.reshape(300, 1)
        X_test = np.hstack((X_test, small_x_new))
out_X_test_path = output_folder + "X_test.npy"
np.save(out_X_test_path, X_test)

Y3_test = np.zeros((3, 1))
for test_count in range(30000, 31000):
    tmp_path = input_y3_path + str(test_count) + ".npy"
    tmp_y3 = np.load(tmp_path)
    tmp_y3 = tmp_y3.T
    Y3_test = np.hstack((Y3_test, tmp_y3))
#Y3_test = Y3_test.T
out_Y3_test_path = output_folder + "Y3_test.npy"
np.save(out_Y3_test_path, Y3_test)

Y8_test = np.zeros((8, 1))
for test_count in range(30000, 31000):
    tmp_path = input_y8_path + str(test_count) + ".npy"
    tmp_y8 = np.load(tmp_path)
    tmp_y8 = tmp_y8.T
    Y8_test = np.hstack((Y8_test, tmp_y8))
#Y8_test = Y8_test.T
out_Y8_test_path = output_folder + "Y8_test.npy"
np.save(out_Y8_test_path, Y8_test)

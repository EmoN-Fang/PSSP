import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections

import time
time_start = time.time()

torch.manual_seed(1)
input_x_path = "/home/emon/Data/blast/100/pataa_npy/"
input_y3_path = "/home/emon/Data/blast/100/new_second_three_npy/"
input_y8_path = "/home/emon/Data/blast/100/new_second_eight_npy/"
output_folder = "/home/emon/Data/blast/100/LSTM_torch/"


X_train = []
for train_count in range(0, 100):
    tmp_path = input_x_path + str(train_count) + ".npy"
    tmp_x = np.load(tmp_path)
    X_train.append(tmp_x)

Y3_train = []
for train_count in range(0, 100):
    tmp_path = input_y3_path + str(train_count) + ".npy"
    tmp_y3 = np.load(tmp_path)
    Y3_train.append(tmp_y3)


Y8_train = []
for train_count in range(0, 100):
    tmp_path = input_y8_path + str(train_count) + ".npy"
    tmp_y8 = np.load(tmp_path)
    Y8_train.append(tmp_y8)

X_test = []
for test_count in range(100, 110):
    tmp_path = input_x_path + str(test_count) + ".npy"
    tmp_x = np.load(tmp_path)
    X_test.append(tmp_x)


Y3_test = []
for test_count in range(100, 110):
    tmp_path = input_y3_path + str(test_count) + ".npy"
    tmp_y3 = np.load(tmp_path)
    Y3_test.append(tmp_y3)

Y8_test = []
for test_count in range(100, 110):
    tmp_path = input_y8_path + str(test_count) + ".npy"
    tmp_y8 = np.load(tmp_path)
    Y8_test.append(tmp_y8)


training_num = len(X_train)
EMBEDDING_DIM = 20
HIDDEN_DIM = 128
MID_DIM = 32
y_size = 3

def mat_to_float_var(matrix):
    V = torch.tensor(matrix)
    V = V.view(V.size()[0], 1, V.size()[1])
    V_new = V.float().cuda()
    return V_new

def mat_to_long_var(matrix):
    V = torch.tensor(matrix).cuda()
    return V

# def to_scalar(var): 
#     # returns a python float
#     return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, mid_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=3, dropout = 0.1, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2mid = nn.Linear(hidden_dim, mid_dim)

        self.mid2tag = nn.Linear(mid_dim, tagset_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(6, 1, self.hidden_dim // 2).cuda(),
                torch.zeros(6, 1, self.hidden_dim // 2).cuda())

    def forward(self, pssm):
        lstm_out, self.hidden = self.lstm(pssm, self.hidden)
        new_input = lstm_out.view(pssm.size()[0], -1)
        mid_space = self.hidden2mid(new_input)
        tag_space = self.mid2tag(mid_space)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, MID_DIM, y_size).cuda()
loss_function = nn.NLLLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8, weight_decay=1e-5)

cost = 0
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    for i in range(len(X_train)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        pssm = mat_to_float_var(X_train[i])
        targets = mat_to_long_var(Y3_train[i])

        #print(sentence_in)

        # Step 3. Run our forward pass.
        tag_scores = model(pssm)
        # print(tag_scores)
        # print(targets)
        # print(tag_scores.shape) 
        # print(targets.shape)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        cost = loss
    if epoch % 10 == 0:
        print(loss)

# See what the scores are after training

model.eval()
final_pred = []
truth_y = []

pred_file = "/home/emon/Data/blast/100/BiLSTM_torch/pred_file.txt"
g = open(pred_file, "w")

with torch.no_grad():
    for i in range(len(X_test)):
        tag_scores = model(mat_to_float_var(X_test[i]))
        # print(tag_scores.size())
        tag_label = argmax(tag_scores).cpu()
        truth = mat_to_long_var(Y3_test[i]).cpu()
        num_tag_lable = tag_label.numpy()
        num_truth = truth.numpy()
        # print(num_truth)
        g.write("> pred_y:"+'\n')
        g.write(str(num_tag_lable))
        g.write('\r\n')
        g.write("> truth_y:"+'\n')
        g.write(str(num_truth))
        g.write('\r\n')

        final_pred = np.hstack((final_pred, num_tag_lable))
        truth_y = np.hstack((truth_y, num_truth))

    # print(final_pred)
    # print(truth_y) 

    accur = np.sum(np.equal(final_pred, truth_y))/truth_y.shape[0]

    print("accur = ", accur)

    freq = collections.defaultdict(lambda :0)
    for i in range(len(truth_y)):
        if final_pred[i] != truth_y[i]:
            freq[truth_y[i]] += 1

    print(freq)
    print(len(truth_y))

g.close()





    # print(tag_scores)
# parameters = model(X_train, Y3_train, X_test, Y3_test, num_class=3)
time_end = time.time()
print('totally time cost', time_end - time_start)

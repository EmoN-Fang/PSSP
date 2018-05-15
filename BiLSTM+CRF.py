import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
input_x_path = "/home/emon/Data/blast/100/pataa_npy/"
input_y3_path = "/home/emon/Data/blast/100/new_second_three_npy/"
input_y8_path = "/home/emon/Data/blast/100/new_second_eight_npy/"
output_folder = "/home/emon/Data/blast/100/LSTM_torch/"


X_train = []
for train_count in range(0, 1000):
    tmp_path = input_x_path + str(train_count) + ".npy"
    tmp_x = np.load(tmp_path)
    X_train.append(tmp_x)

Y3_train = []
for train_count in range(0, 1000):
    tmp_path = input_y3_path + str(train_count) + ".npy"
    tmp_y3 = np.load(tmp_path)
    Y3_train.append(tmp_y3)


Y8_train = []
for train_count in range(0, 1000):
    tmp_path = input_y8_path + str(train_count) + ".npy"
    tmp_y8 = np.load(tmp_path)
    Y8_train.append(tmp_y8)

X_test = []
for test_count in range(1000, 1100):
    tmp_path = input_x_path + str(test_count) + ".npy"
    tmp_x = np.load(tmp_path)
    X_test.append(tmp_x)


Y3_test = []
for test_count in range(1000, 1100):
    tmp_path = input_y3_path + str(test_count) + ".npy"
    tmp_y3 = np.load(tmp_path)
    Y3_test.append(tmp_y3)

Y8_test = []
for test_count in range(1000, 1100):
    tmp_path = input_y8_path + str(test_count) + ".npy"
    tmp_y8 = np.load(tmp_path)
    Y8_test.append(tmp_y8)




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


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    # print("$$$$vec=",vec)
    # print("hello-vec=",vec[0])
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True)

        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))


        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000


        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(4, 1, self.hidden_dim // 2).cuda(),
                torch.zeros(4, 1, self.hidden_dim // 2).cuda())

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # print("init_alphas= ",init_alphas)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        # print("init_alphas= ",init_alphas)
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # print("～～～～～～～～～～～～～～～～～～～～～～features=",feats)
        # print("features.size=",feats.size())   #(67,5)
        # Iterate through the sentence
        for feat in feats:
            # print("feat=",feat)
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # print("next_tag =", next_tag)
                # print("trans_score=", trans_score)
                # print("emit_score= ", emit_score)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var.cuda() + trans_score.cuda() + emit_score.cuda()
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                # print("alphas_t=", alphas_t)
            forward_var = torch.cat(alphas_t).view(1, -1)
            # print("forward_var=", forward_var)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        # print("*******terminal_var=",terminal_var)  #(1,5)
        # print("~~~~~~~~~transitioningons=", self.transitions)
        # self.transitions = torch.div(self.transitions, 10)
        return alpha



    def _get_lstm_features(self, pssm):
        # print("pssm=",pssm)
        # print("pssm.size=",pssm.size())
        # self.hidden = self.init_hidden()
        # print("pssm.size= ", pssm.size())
        lstm_out, self.hidden = self.lstm(pssm, self.hidden)

        lstm_out = lstm_out.view(pssm.size()[0], -1)
        # print("--------")
        # print("lstm_out=",lstm_out)
        # print("lstm_out.size=",lstm_out.size())   #(67,128)
        lstm_feats = self.hidden2tag(lstm_out)     
        return lstm_feats       #(67,5)

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).cuda()
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).cuda(), tags])
        # print(tags.size())
        # print(feats.size())
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars.cuda()
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        # print("f_score=",forward_score)
        # print("g_score=",gold_score)
        # print("----------delta=",forward_score - gold_score)
        return (forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)

        # print("-------------------------------------hello-im-score:",score)
        # print("-------------------------------------hello-im-tag_seq:",tag_seq)
        return score, tag_seq


training_num = len(X_train)
EMBEDDING_DIM = 20
HIDDEN_DIM = 64
START_TAG = "<START>"
STOP_TAG = "<STOP>"


tag_to_ix = {"0": 0, "1": 1, "2": 2, START_TAG: 3, STOP_TAG: 4}
# tag_to_ix = {"0": 0, "1": 1, "2": 2, "3":3, "4":4, "5":5, "6":6, "7":7, START_TAG: 8, STOP_TAG: 9}



model = BiLSTM_CRF(tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8, weight_decay=1e-6)


cost = 0
for epoch in range(600):  # again, normally you would NOT do 300 epochs, it is toy data
    for i in range(len(X_train)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        model.hidden = model.init_hidden()
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        pssm = mat_to_float_var(X_train[i])
        targets = mat_to_long_var(Y3_train[i])
        # print("X_train=",pssm)
        # print("X_trainSIZE=",pssm.size())

        #print(sentence_in)

        loss = model.neg_log_likelihood(pssm, targets).cuda()
        if epoch % 20 == 0:
            print("loss",i,"=",loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1e-1)
        optimizer.step()

        cost = loss
    # if epoch % 10 == 0:
    #     print(loss)

# See what the scores are after training

final_pred = []
truth_y = []

with torch.no_grad():
    for i in range(len(X_test)):
        tag_scores = model(mat_to_float_var(X_test[i]))
        n_tag_scores = np.array([])
        for item in tag_scores[1]:
            n_tag_scores = np.append(n_tag_scores, [item[0].cpu().numpy()])
        n_tag_scores.astype(int)

        truth = mat_to_long_var(Y3_test[i]).cpu()
        num_truth = truth.numpy()

        final_pred = np.hstack((final_pred, n_tag_scores))
        truth_y = np.hstack((truth_y, num_truth))

    print("fianl_pred=",final_pred)
    print("truth_y=",truth_y)

    accur = np.sum(np.equal(final_pred, truth_y))/truth_y.shape[0]

    print(accur)

    # print(tag_scores)

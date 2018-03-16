# PSSP
my graduation project - Protein Secondary Structure Prediction


9 weeks left
2 weeks for writing paper
6 weeks for training models
1 week for data processing 

There's 5 types of prediction

1. 1*20 input - 1 output, 1d CNN      --- use PSSM_1. PSSM_2
2. 7*20 input - 1 output, regular pssm window    --- USE PSSM_1, PSSM_2, ONE-HOT
4. n*20 input - n output, ResNet??? but size is not identical    ---USE PSSM_1, PSSM_2
5. n*20 input - n output, Bidirection LSTM   ---USE PSSM1, PSSM2, ONE-HOT  


train 3 lr model???  this is naive
how to classify 3 label?

check summer project for embedding
check CS231n for n label classify
learn Ng

one-hot
PSSM_1
PSSM_2


next week:
process the pssm matrix, figure out a way to store the encoder.  seperately or in a how .npv?
Replace the original sequence with encorder, then store in a matrix, like independ images?
Or just store the sequences and replace them while inputting?
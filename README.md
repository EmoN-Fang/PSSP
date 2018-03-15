# PSSP
my graduation project - Protein Secondary Structure Prediction


9 weeks left
2 weeks for writing paper
6 weeks for training models
1 week for data processing 

There's 5 types of prediction

1. 9*1 input - 1 output, very basic regrussion
2. 7*20 input - 1 output, regular pssm window 
3. 1*20 input - 1 output, 1d CNN
4. n*20 input - n*20 output, ResNet??? but size is not identical
5. n*20 input - n output, Bidirection LSTM


next week:
process the pssm matrix, figure out a way to store the encoder.  seperately or in a how .npv?
Replace the original sequence with encorder, then store in a matrix, like independ images?
Or just store the sequences and replace them while inputting?
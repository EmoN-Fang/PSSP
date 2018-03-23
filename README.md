# PSSP
my graduation project - Protein Secondary Structure Prediction


9 weeks left
2 weeks for writing paper
6 weeks for training models
1 week for data processing 


!!! NO NEED FOR Embidding
0.choose all the 200-/500-/1000- data
1.process them into .fasta one by one, 
2.then using 
    "import os  
    for i in range(0,65):  
    os.system("tar -xzvf nr.%02d.tar.gz"%i)"
    cd Data/blast
    psiblast -db pdbaa -query test.fasta -num_iterations 6 -evalue 0.001 -num_threads 16 -out_ascii_pssm pssm1.pssm -out pssm1.txt
to make them into pssm#.pssm
3.process .pssm, till we get the matrix for each sequence
4.send the matrix directly in the the network


embidding only for one-hot, and using bilstm, we can have a try :)


There's 4 types of prediction

1. 1*20 input - 1 output, 1d CNN/LR      --- use PSSM_1. PSSM_2
2. 7*20 input - 1 output, regular pssm window, 2D CNN/LR    --- USE PSSM_1, PSSM_2, ONE-HOT
3. n*20 input - n output, ResNet??? but size is not identical, pixel to pixel?    ---USE PSSM_1, PSSM_2
4. n*20 input - n output, Bidirection LSTM   ---USE PSSM1, PSSM2, ONE-HOT  


train 3 lr model???  this is naive
how to classify 3 label?  check CS231n for n label classify,softmax

check summer project for embedding  ,pytorch
learn Ng

one-hot
PSSM_1
PSSM_2


next week:
process the pssm matrix, figure out a way to store the encoder.  seperately or in a how .npv?
Replace the original sequence with encorder, then store in a matrix, like independ images?
Or just store the sequences and replace them while inputting?
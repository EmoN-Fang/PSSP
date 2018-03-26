import os
for i in range(0, 23303):
    os.system("psiblast -db pataa -query /50/input_seq/%d.fasta -num_iterations 3 -evalue 0.001 -num_threads 32 -out_ascii_pssm 50/pataa/%d.pssm -out 50/pataa/%d.txt" % (i,i,i))
    # os.system("psiblast -db pdbaa -query input_seq/%d.fasta -num_iterations 3 -evalue 0.001 -num_threads 32 -out_ascii_pssm pdbaa/%d.pssm -out pdbaa/%d.txt" % (i,i,i))
    # os.system("psiblast -db swissprot -query input_seq/%d.fasta -num_iterations 3 -evalue 0.001 -num_threads 32 -out_ascii_pssm swissprot/%d.pssm -out swissprot/%d.txt" % (i,i,i))
    #os.system("tar -xzvf nr.%02d.tar.gz" % i)

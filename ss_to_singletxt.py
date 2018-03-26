import numpy as np

f = open("ss.txt", "r")
# print(f.read())
get = f.read()
result = get.split('>')
# "".join(result.split())

f.close()

j = 0
for i in range(1, len(result)):
    # "".join(result[i].split( ))
    # re.split(r'\s+', result[i])
    result[i] = result[i].replace('\n', '')
    if (i % 2 == 1):
        simple_seq = result[i][15:]
        if len(simple_seq)<51:
                g = open("50/input_seq/"+str(j)+".fasta","w")
                g.write(">"+result[i][:15]+'\n')
                g.write(simple_seq)
                g.close()
    else:
        second_eight = result[i][13:]
        if len(second_eight)<51:
            g = open("50/second_eight/"+str(j)+".txt","w")
            g.write(">"+result[i][:13]+'\n')
            g.write(second_eight)
            g.close()
            j += 1

j = 0

for i in range(2, len(result), 2):
    # "".join(result[i].split( ))
    # re.split(r'\s+', result[i])
    result[i] = result[i].replace('\n', '')
    result[i] = result[i].replace('G', 'H')
    result[i] = result[i].replace('I', 'H')
    result[i] = result[i].replace('B', 'E')
    result[i] = result[i].replace('S', 'C')
    result[i] = result[i].replace('T', 'C')
    second_three = result[i][13:]
    if len(second_three)<51:
            g = open("50/second_three/"+str(j)+".txt","w")
            g.write(">"+result[i][:13]+'\n')
            g.write(second_three)
            g.close()
            j += 1

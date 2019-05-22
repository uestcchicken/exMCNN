from network import MCNN
import sys

if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('usage: python3 train.py A(or B)')
    exit()

EPOCH = 2000
            
mcnn = MCNN(dataset)
mcnn.train(EPOCH)










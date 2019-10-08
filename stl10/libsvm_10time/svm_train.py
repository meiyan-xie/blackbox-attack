import numpy as np
import random
from sklearn.datasets import dump_svmlight_file
from subprocess import call
import torchvision.datasets as datasets

'''

'/research/urextra/usman/liblinear-multicore-2.20/train -s 2 -B 1 -c 1 -n 8 traindata'

## Comment to call
../../liblinear-multicore-2.20/train -s 2 -B 1 -c 1 -n 8 <train file>
<model file>
../../liblinear-multicore-2.20/predict <test file> <model file> <predictions>

'''

def bootstrapping():
    # data_dir = '/research/datasci/mx42/stl10_1000/stl10_1k_new'
    # train = datasets.STL10(root=data_dir, split='train', download=False, transform=None)

    # train_x = train.data.reshape(-1, 3*96*96)
    # train_y = train.labels

    # return train_x, train_y

    train_dir = '/research/datasci/mx42/stl10-original/01loss/'
    train_file = 'stl10.train'

    traindataset = np.loadtxt(train_dir + train_file)
    traindata = traindataset[:, 1:]
    trainlabel = traindataset[:, :1]
    trainlabel = trainlabel.flatten()

    dump_svmlight_file(traindata, trainlabel, 'stl10_lib', zero_based=False)

    # return traindata, trainlabel


# External call liblinear to train model
def external_call(model):
    print('External call')
    traindata = 'stl10_lib'
    cmd = '/research/urextra/usman/liblinear-multicore-2.20/train'

    s = 2
    B = 1
    n = 8
    c = 1

    call([cmd, '-s', str(s), '-B', str(B), '-c', str(c), '-n', str(n), traindata, model])


def train():
    i = 0
    while i < 10:
        print('\niter ', i)
        model = 'model/model_' + str(i)
        bootstrapping()
        external_call(model)
        i += 1

train()

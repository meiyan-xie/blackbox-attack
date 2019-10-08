import numpy as np
import random
from sklearn.datasets import dump_svmlight_file
from subprocess import call

'''

'/research/urextra/usman/liblinear-multicore-2.20/train -s 2 -B 1 -c 1 -n 8 traindata'

## Comment to call
../../liblinear-multicore-2.20/train -s 2 -B 1 -c 1 -n 8 <train file>
<model file>
../../liblinear-multicore-2.20/predict <test file> <model file> <predictions>

'''


# External call liblinear to train model
def external_call(model):
    print('External call')
    data = np.loadtxt('cifar_traindata')
    label = np.loadtxt('cifar_trainlabels')

    dump_svmlight_file(data, label, 'cifar_traindata_lib', zero_based=False)

    traindata = 'cifar_traindata_lib'

    cmd = '/research/urextra/usman/liblinear-multicore-2.20/train'

    s = 2
    B = 1
    n = 8
    c = 1
    call([cmd, '-s', str(s), '-B', str(B), '-c', str(c), '-n', str(n), traindata, model])


def main():
    model = 'model_1time_s2'
    external_call(model)

main()

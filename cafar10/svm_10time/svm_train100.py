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


# Generate bootstrapping data and dump to liblinear format
def bootstrapping():
    print('Do bootstrapping')
    # Load features and labels
    features = np.loadtxt('cifar_traindata')
    labels = np.loadtxt('cifar_trainlabels')

    # Concatenate labels with features
    data = np.concatenate((labels.reshape(len(labels), 1), features), axis=1)

    n_row = len(data)

    # Randomly select 1 row and repeat n times.
    newdata = []
    for i in range(n_row):
        idx = random.randint(0, n_row - 1)
        newdata.append(data[idx])

    newdata = np.array(newdata)
    new_features = newdata[:,1:]
    new_labels = newdata[:,0]

    print('Dump_svmlight_file')
    # Convert normal text format to liblinear format
    dump_svmlight_file(new_features, new_labels, 'traindata', zero_based=False)


# External call liblinear to train model
def external_call(model):
    print('External call')
    traindata = 'traindata'
    cmd = '/research/urextra/usman/liblinear-multicore-2.20/train'

    s = 2
    B = 1
    n = 8
    c = 1
    call([cmd, '-s', str(s), '-B', str(B), '-c', str(c), '-n', str(n), traindata, model])


# Run liblinear svm n times
def run_svm100times():
    i = 0
    while i < 100:
        print('\niter ', i)
        model = 'model/model_' + str(i)
        bootstrapping()
        external_call(model)
        i += 1

def main():
    # Run one time
    bootstrapping()
    model = 'model0'
    external_call(model)

# main()


run_svm100times()

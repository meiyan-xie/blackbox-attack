
import numpy as np
import random
from subprocess import call
from sklearn.datasets import dump_svmlight_file


'''

'/research/urextra/usman/liblinear-multicore-2.20/train -s 2 -B 1 -c 1 -n 8 traindata'

## Comment to call
../../liblinear-multicore-2.20/train -s 2 -B 1 -c 1 -n 8 <train file>
<model file>
../../liblinear-multicore-2.20/predict <test file> <model file> <predictions>

'''


# External call liblinear to train model
def external_call(testdata_file):

    features = np.loadtxt(testdata_file)

    # Generate pseudo labels
    labels = []
    for i in range(0,len(features)):
        x = random.randint(0, 9)
        labels.append(x)
    labels = np.array(labels)

    # Convert label and features into libsvm format and save in a file
    dump_svmlight_file(features, labels, 'testdata_inter_s0', zero_based=False)

    testdata = 'testdata_inter_s0'

    print('External call')

    model = 'model_1time_s0'
    prediction = 'predicted_inter'
    cmd = '/research/urextra/usman/liblinear-multicore-2.20/predict'
    call([cmd, testdata, model, prediction])


def main(testdata_file):
    external_call(testdata_file)
    predicted_y = np.loadtxt('predicted_inter_s0')

    return predicted_y

if __name__ == '__main__':
    testdata_file = 'cifar_testdata'
    main(testdata_file)

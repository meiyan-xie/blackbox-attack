
import numpy as np
import random
from subprocess import call
import torchvision.datasets as datasets
from sklearn.datasets import dump_svmlight_file


'''

'/research/urextra/usman/liblinear-multicore-2.20/train -s 2 -B 1 -c 1 -n 8 traindata'

## Comment to call
../../liblinear-multicore-2.20/train -s 2 -B 1 -c 1 -n 8 <train file>
<model file>
../../liblinear-multicore-2.20/predict <test file> <model file> <predictions>

'''


def get_data():
    # data_dir = '/research/datasci/mx42/stl10_1000/stl10_1k_new'
    # test = datasets.STL10(root=data_dir, split='test', download=False, transform=None)

    # test_x = test.data.reshape(-1, 3*96*96)
    # test_y = test.labels

    # return test_x, test_y

    test_dir = '/research/datasci/mx42/stl10-original/01loss/'
    test_file = 'stl10.test'

    testdataset = np.loadtxt(test_dir + test_file)
    testdata = testdataset[:, 1:]
    testlabel = testdataset[:, :1]

    return testdata, testlabel


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
    dump_svmlight_file(features, labels, 'testdata_inter_2', zero_based=False)

    testdata = 'testdata_inter_2'

    print('External call')
    model = 'model/libsvm_trained_2'
    prediction = 'predicted_inter_2'
    cmd = '/research/urextra/usman/liblinear-multicore-2.20/predict'
    call([cmd, testdata, model, prediction])


def predict(testdata_file):
    external_call(testdata_file)
    predicted_y = np.loadtxt('predicted_inter_2')


    return predicted_y

if __name__ == '__main__':
    testdata_file = 'stl10'
    predict(testdata_file)

# SVM predict
import numpy as np
from subprocess import call
from sklearn.datasets import dump_svmlight_file
import torchvision.datasets as datasets

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
    testlabel = testlabel.flatten()

    return testdata, testlabel

features, labels = get_data()

dump_svmlight_file(features, labels, 'test_lib_2', zero_based=False)

testdata = 'test_lib_2'
model = '/research/datasci/mx42/stl10-original/libsvm/model/libsvm_trained_2'
prediction = 'predicted_test_2'
cmd = '/research/urextra/usman/liblinear-multicore-2.20/predict'
call([cmd, testdata, model, prediction])

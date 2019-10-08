import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def load_data():
    print('load data')
    dir = '/Users/meiyan/Desktop/blackbox/blackbox_minst/attack_svc/'
    traindata_filename = '200traindata'
    trainlabel_filename = '200trainlabels'
    testdata_filename = '20testdata'
    testlabel_filename = '20testlabel'

    traindata = np.loadtxt(dir + traindata_filename)
    trainlabel = np.loadtxt(dir + trainlabel_filename)
    testdata = np.loadtxt(dir + testdata_filename)
    testlabel = np.loadtxt(dir + testlabel_filename)

    return traindata, trainlabel, testdata, testlabel


def train_svm(traindata, trainlabel, testdata, testlabel):

    clf = LinearSVC(C=1)

    # ! Most time-comsuming
    print('trainning...')
    clf.fit(traindata, trainlabel)
    predicted_y = clf.predict(testdata)

    print(predicted_y)


def main():

    traindata, trainlabel, testdata, testlabel = load_data()
    train_svm(traindata, trainlabel, testdata, testlabel)

main()

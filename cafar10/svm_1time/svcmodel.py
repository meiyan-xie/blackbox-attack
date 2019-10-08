import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# For saving / loading model
from sklearn.externals import joblib



def load_data():
    print('load data')
    dir = '/research/datasci/mx42/01loss/01loss/'
    traindata_filename = 'cifar_traindata'
    trainlabel_filename = 'cifar_trainlabels'
    testdata_filename = 'test_adv_epoch2'
    testlabel_filename = 'test_label_for_epoch2'

    traindata = np.loadtxt(dir + traindata_filename)
    trainlabel = np.loadtxt(dir + trainlabel_filename)
    testdata = np.loadtxt(dir + testdata_filename)
    testlabel = np.loadtxt(dir + testlabel_filename)

    return traindata, trainlabel, testdata, testlabel


def train_svm(traindata, trainlabel, testdata, testlabel):

    # for c in c_values:
    clf = LinearSVC(C=1, dual=False)

    # ! Most time-comsuming
    print('trainning...')
    clf.fit(traindata, trainlabel)

    print('predict test data')
    predicted = clf.predict(testdata)
    accuracy = accuracy_score(testlabel, predicted)
    print('Test accuracy: ', accuracy)

    ##### 10 Fold cross validation
    # print('10 fold cv')
    # scores = cross_val_score(clf, traindata, trainlabel, cv = 10)
    # train_accuracy = scores.mean()
    # print('C ', c, 'train_accuracy ', train_accuracy)

    # # Save model
    # print('Saving the trained model')
    # joblib.dump(clf, 'svcmodel.pkl')



def main():

    traindata, trainlabel, testdata, testlabel = load_data()
    train_svm(traindata, trainlabel, testdata, testlabel)

main()

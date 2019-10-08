import numpy as np
from sklearn.metrics import accuracy_score
# For saving / loading model
from sklearn.externals import joblib


dir = '/research/datasci/mx42/01loss/01loss/'
testdata_filename = 'test_adv_epoch2'
testlabel_filename = 'test_label_for_epoch2'
testdata = np.loadtxt(dir + testdata_filename)
testlabel = np.loadtxt(dir + testlabel_filename)


# Load model
clf = joblib.load('svcmodel.pkl')
predicted = clf.predict(testdata)

accuracy = accuracy_score(testlabel, predicted)
print('Test accuracy on adv test data: ', accuracy)

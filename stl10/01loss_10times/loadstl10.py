import numpy as np
def get_data():
    test_dir = '/research/datasci/mx42/stl10-original/01loss_test/'
    testdata_filename = 'stl10.test'
    predicted = 'predicted'
    testdataset = np.loadtxt(test_dir + testdata_filename)
    testdata = testdataset[:, 1:]
    testlabel = testdataset[:, :1]

    # print(len(testdataset[0]))
    # print(len(testdata[0]))
    # print(len(testlabel[0]))
    # print(testlabel[:5])
    count = 0
    predicted_y = np.loadtxt(test_dir + predicted)
    for i in range(len(predicted_y)):
        if predicted_y[i] == testlabel[i]:
            count += 1

    accuracy = count / len(predicted_y)
    print('accuracy: ', accuracy)


get_data()

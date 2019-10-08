import numpy as np

truelabel_file = 'cifar_testlabels'
predicted_label_file = 'out'

truelabel = np.loadtxt(truelabel_file)
predictedlabel = np.loadtxt(predicted_label_file)

n = len(truelabel)
print(n)
print(len(predictedlabel))

count = 0
for i in range(n):
    if truelabel[i] == predictedlabel[i]:
        count += 1
print('count', count)
accuracy = count / n
print('accuracy: ', accuracy)

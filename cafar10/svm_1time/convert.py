from sklearn.datasets import dump_svmlight_file
import numpy as np
import random

features = np.loadtxt('cifar_testdata')
labels = np.loadtxt('cifar_testlabels')

# # Generate pseudo labels
# labels = []
# for i in range(0,len(features)):
#     x = random.randint(0, 9)
#     labels.append(x)
# labels = np.array(labels)

dump_svmlight_file(features, labels, 'cifar_testdata_lib', zero_based=False)

import random
import numpy as np

data = np.loadtxt('5testdata')
newdata = []
for i in range(len(data)):
    idx = random.randint(0, len(data) -1)
    newdata.append(data[idx])

print(newdata)

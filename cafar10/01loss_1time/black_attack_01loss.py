import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import LinearModel
import torch.backends.cudnn as cudnn
import time
import torch.optim as optim
import os
import sys
import subprocess


class Target(object):

    def predict(self, x_sub):
        # 1. Convert input x (tensor) to numpy array
        # 2. Convert 0-1 value to 0-255 value
        # 3. Save result to a text file (01loss's input is a file)
        x_sub_np = x_sub.numpy()
        x_sub_np = x_sub_np * 255
        x_sub_np = x_sub_np.astype(int)
        np.savetxt('test_sub_0', x_sub_np, fmt='%d')

        # cmd = perl predict_01loss.pl cifar_testdata mapping_test
        predicted_y = subprocess.check_output(['perl', '/research/datasci/mx42/attack_cifar10/01loss/predict_01loss.pl', 'test_sub_0', '/research/datasci/mx42/attack_cifar10/01loss/mapping_test'])

        # predicted_y is a byte, need to convert to a string, than convert to a numpy array
        predicted_y = predicted_y.decode()

        predicted_y = np.fromstring(predicted_y, dtype=int, sep='\n')

        # Convert numpy array to tensor (long type)
        torchTensor_y = torch.from_numpy(predicted_y)
        # print('torchTensor_y\n', torchTensor_y)

        return torchTensor_y.long()

    def eval(self, testdata, testlabel):
        # truelabel is a torch tensor
        predicted_y = self.predict(testdata)
        n_predicted = len(predicted_y)

        correct_count = 0
        correct_count += predicted_y.eq(testlabel).sum().item()
        print('correct_count: ', correct_count)
        accuracy = correct_count / n_predicted
        print('Test accuray tensor: ', accuracy)


class Substitute(object):

    def __init__(self, model, save_path='None', device=None):
        self.device = device
        self.model = model
        self.save_path = save_path
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(self.save_path)['net'])
            print('Load weights successfully for %s' % self.save_path)
        else:
            print('Initialized weights')
        self.model.to(device=device)

    # Get input data
    def get_loader(self, x=None, y=None, batch_size=100, shuffle=False):
        assert isinstance(x, torch.Tensor)
        if y is None:
            y = torch.full(size=(x.size(0),), fill_value=-1).long()
        dataset = torch.utils.data.TensorDataset(x, y)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=2, pin_memory=True)

    def predict(self, x, batch_size):
        self.get_loader(x, batch_size=batch_size, shuffle=False)
        self.model.eval()
        pred = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data = data.to(device=self.device, dtype=dtype)

                outputs = self.model(data)
                pred.append(outputs.data.max(1)[1])

        # Convert gup data datatype to cpu datatype
        return torch.cat(pred).cpu()

    def eval(self, x, y, batch_size):
        self.get_loader(x, y, batch_size=batch_size, shuffle=False)
        self.model.eval()

        correct = 0
        a = time.time()

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(self.data_loader):

                data, target = data.to(device=self.device, dtype=dtype), target.to(device=self.device)

                predicted = self.model(data).max(1)[1]
                correct += predicted.eq(target.data).sum().item()

            print('Test_accuracy: %0.5f' % (correct / len(self.data_loader.dataset)))

    def train(self, x, y, batch_size, n_epoch):
        self.get_loader(x, y, batch_size, True)
        self.model.train()

        optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=0.001,
                )
        criterion = nn.CrossEntropyLoss().to(device=self.device)
        for epoch in range(n_epoch):
            train_loss = 0
            correct = 0
            a = time.time()

            for batch_idx, (data, target) in enumerate(self.data_loader):

                data, target = data.to(device=self.device, dtype=dtype), target.to(device=self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = outputs.data.max(1)[1]
                correct += predicted.eq(target.data).sum().item()

    def get_grad(self, x, y):
        self.get_loader(x, y, batch_size=1, shuffle=False)
        grads = []
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(device=self.device, dtype=dtype).requires_grad_()
            outputs = self.model(data)[0, target]
            outputs.backward()
            grads.append(data.grad.cpu())

        return torch.cat(grads, dim=0)

    def get_loss_grad(self, x, y):
        self.get_loader(x, y, batch_size=100, shuffle=False)
        grads = []

        self.model.train()
        criterion = nn.CrossEntropyLoss().to(device=self.device)
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(device=self.device, dtype=dtype).requires_grad_(),\
                           target.to(device=self.device)
            outputs = self.model(data)
            loss = criterion(outputs, target)
            loss.backward()
            grads.append(data.grad.cpu())
        return torch.cat(grads, dim=0)

def get_data():
    data_dir = '/research/datasci/mx42/attack_cifar10/01loss/'
    testdata_filename = 'cifar_testdata'
    testlable_filename = 'cifar_testlabels'

    # Load test data into numpy ndarray
    testdata_np = np.loadtxt(data_dir + testdata_filename)
    # Convert numpy ndarray to torch tensor
    testdata_tensor = torch.from_numpy(testdata_np)

    # Load test label into numpy ndarray
    testlabel_np = np.loadtxt(data_dir + testlable_filename)
    # Convert to torch tensor
    testlabel_tensor = torch.from_numpy(testlabel_np)

    indices = np.random.permutation(10000)

    '''
    # -1 auto calculate sample's number
    # Convert value range from 0-255 to 0-1 (pytorch need 0-1 value range)
    '''
    # sub_x = testdata_tensor[indices[:200]].float().reshape((-1, 3, 32, 32))
    sub_x = testdata_tensor[indices[:200]].float()
    sub_x /= 255

    test_data = testdata_tensor[indices[200:]].float().reshape((-1, 3, 32, 32))
    test_data /= 255
    test_label = testlabel_tensor[indices[200:]]


    # ######## Save test_data for ploting image, just for testing  #########
    # num = test_data.size(0)
    # print('Saving test_data after shuffling')
    # test_data_tensor = test_data.reshape(num, 3*32*32)
    # test_data_np = test_data_tensor.numpy()
    # test_data_np = test_data_np * 255
    # test_data_np = test_data_np.astype(int)
    # np.savetxt('test_data_01loss_2', test_data_np, fmt='%d')

    # ######## Save test_label, just for later usage  #########
    # num = test_data.size(0)
    # print('Saving test_label after shuffling')
    # test_label_np = test_label.numpy()
    # test_label_np = test_label_np.astype(int)
    # np.savetxt('test_label_01loss_2', test_label_np, fmt='%d')
    # ################


    '''
    # sub_x: 200 input to the target model and get y_predicted
    # test_data: the rest 19800 testdata for later testing.
    # test_label: the rest 19800 testdata label for eval
    '''
    return sub_x, test_data, test_label.long()

def jacobian_augmentation(model, x_sub, y_sub, Lambda, samples_max):
    Lambda = np.random.choice([-1, 1])* Lambda
    x_sub_grads = model.get_grad(x=x_sub, y=y_sub)

    x_sub_new = x_sub + Lambda * torch.sign(x_sub_grads)

    # len_x_sub = len(x_sub)
    # # print('x_sub 1: ', len_x_sub)

    # random_seed = 2019
    # indices = list(range(len_x_sub))
    # # print('indices: ,', indices)

    # np.random.seed(random_seed)
    # np.random.shuffle(indices)
    # # print(indices)

    # x_sub = x_sub[indices[:100]]
    # # print('x_sub 2: ', len(x_sub))

    if x_sub.size(0) <= samples_max / 2:
        return torch.cat([x_sub, x_sub_new], dim=0)
    else:
        return x_sub_new

def get_adv(model, x, y, epsilon):
    print('getting grads on epsilon=%.4f'%epsilon)
    grads = model.get_loss_grad(x, y)
    print('generating adversarial examples')
    return (x + epsilon * torch.sign(grads)).clamp_(0, 1)


# Black box algorigthm (see detail in the paper)
def MNIST_bbox_sub(param, target_model, substitute_model, x_sub, test_data, \
                   test_label, aug_epoch, samples_max, n_epoch, fixed_lambda):

    for rho in range(aug_epoch):
        print('Epoch #%d:'%rho)
        # Get x_sub's labels
        print('Current x_sub\'s size is %d'%(x_sub.size(0)))
        a = time.time()

        '''
        # y_sub get target model's output (predicted label)
        # x_sub: at the first time only have 150 (or 200) testdata, after augumentation, number increase
        '''
        # y_sub = oracle_model.predict(x=x_sub, batch_size=oracle_size)
        y_sub = target_model.predict(x_sub)
        print('Get label for x_sub cost %.1f'%(time.time() - a))

        # Train substitute model
        substitute_model.train(x=x_sub, y=y_sub, batch_size=128, n_epoch=n_epoch)

        if rho < param['data_aug'] - 1:
            print('Substitute data augmentation processing')
            a = time.time()
            x_sub = jacobian_augmentation(model=substitute_model, x_sub=x_sub, y_sub=y_sub, \
                                          Lambda=fixed_lambda, samples_max=samples_max)

            print('Augmentation cost %.1f seconds'%(time.time() - a))

        #Generate adv examples
        test_adv = get_adv(model=substitute_model, x=test_data, y=test_label, epsilon=param['epsilon'])

        # Compute the accuracy adversarial samples on target model
        # if rho % 2 == 0:
        # if rho == 0:

        n_sample = test_adv.size(0)

            # ####  Visualize the image and see the different  ####
            # # Save the adversarial examples
            # print('Saving test adv data')
            # test_adv_tensor = test_adv.reshape(n_sample, 3*32*32)
            # test_adv_np = test_adv_tensor.numpy()
            # test_adv_np = test_adv_np * 255
            # test_adv_np = test_adv_np.astype(int)
            # np.savetxt('test_adv_epoch2', test_adv_np, fmt='%d')
        if (rho > 19 and rho % 2 == 0) or rho == aug_epoch - 1:
            print('Oracle model FGSM attack\'s accuracy on adversarial samples #%d:' % (n_sample))
            target_model.eval(test_adv.reshape(n_sample, 3*32*32), test_label)
            torch.save(substitute_model.model.state_dict(), 'model/sub_01loss_0.t7')

if __name__ == '__main__':
    param = {
        'hold_out_size': 150,
        'test_batch_size': 128,
        'nb_epochs': 10,
        'learning_rate': 0.001,
        'data_aug': 40,
        # 'oracle_name': 'model/lenet',
        'epsilon': 0.0625,
        'lambda': 0.1,  # In the paper it is 0.1, but in our early experiment for 01loss and svm attack, we use 0.0625.
        # 'lambda': 0.0625,
    }

    global seed, dtype, oracle_size
    oracle_size = 20
    dtype = torch.float32
    device = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
    seed = 2018
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    sub_x, test_data, test_label = get_data()

    target_model = Target()
    n_sample = test_data.size(0)
    print('Oracle model evaluation on clean data #%d:'%(n_sample))
    target_model.eval(test_data.reshape(n_sample, 3*32*32), test_label)

    sub = LinearModel()

    substitute_model = Substitute(model=sub, device=device)

    MNIST_bbox_sub(param=param, target_model=target_model, substitute_model=substitute_model, \
                   x_sub=sub_x, test_data=test_data, test_label=test_label, aug_epoch=param['data_aug'],\
                   samples_max=12800, n_epoch=param['nb_epochs'], fixed_lambda=param['lambda'])

    print('\n\nFinal results:')
    # target_model.eval(test_data.reshape(n_sample, 3*32*32), test_label)
    print('Substitute model evaluation on clean data: #%d:'%(n_sample))
    substitute_model.eval(x=test_data, y=test_label, batch_size=512)

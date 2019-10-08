import numpy as np
import matplotlib.pyplot as plt


def plot_image(data, idx, datatype):
    plt.figure(figsize=(0.8, 0.8))
    x = data[idx]
    r = x[:1024].reshape(32, 32)
    g = x[1024:2048].reshape(32, 32)
    b = x[2048:].reshape(32, 32)
    img = np.dstack((r, g, b))
    plt.imshow(img)
    fn_save = datatype + '.png'
    plt.savefig(fn_save, bbox_inches='tight')


def main():
    print('loading data')
    dir = '/Users/meiyan/Desktop/blackbox/blackbox_minst/attack_svc/'
    filename_org = 'test_data_svc'
    filename_adv = 'test_adv_svc'
    test_org = np.loadtxt(dir + filename_org, dtype=int)
    test_adv = np.loadtxt(dir + filename_adv, dtype=int)
    idx = 5
    print('Plot original figure...')
    plot_image(test_org, idx, 'test_org')
    print('Plot modified figure...')
    plot_image(test_adv, idx, 'test_adv')

main()

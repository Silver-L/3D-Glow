import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d

class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        self.f_log = open(path, 'w')
        self.f_log.write(json.dumps(kwargs) + '\n')

    def log(self, **kwargs):
        self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()


def visualize_slices(X, Xe, outdir):
    # plot reconstruction
    fig, axes = plt.subplots(ncols=10, nrows=2, figsize=(18, 4))
    for i in range(10):
        minX = np.min(X[i, :])
        maxX = np.max(X[i, :])
        axes[0, i].imshow(X[i, :].reshape(8, 8), cmap=cm.Greys_r, vmin=minX, vmax=maxX,
                          interpolation='none')
        axes[0, i].set_title('original %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        minXe = np.min(Xe[i, :])
        maxXe = np.max(Xe[i, :])
        axes[1, i].imshow(Xe[i, :].reshape(8, 8), cmap=cm.Greys_r, vmin=minXe, vmax=maxXe,
                          interpolation='none')
        axes[1, i].set_title('reconstruction %d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "/reconstruction.png")
    # plt.show()

def interpolation(y, z1, z2, model):
    x1_2 = []
    for a in [0.1*x for x in range(11)]:
        z1_2 = (1.0 - a) * z1 + a * z2
        x1_2.append(model.decode(y, z1_2))
    x1_2 = np.concatenate(x1_2, axis=0)
    return(x1_2)

def display_slices(case, size, num_data, outdir):
    # case: image data, num_data: number of data, size: length of a side
    min = np.min(case)
    max = np.max(case)
    # axial
    fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
    for i in range(size):
        for j in range(num_data):
            axes[j, i].imshow(case[j, i, :, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[j, i].set_title('z = %d' % i)
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "/interpolation.png")

def display_center_slices(case, size, num_data, outdir):
    # case: image data, num_data: number of data, size: length of a side
    min = np.min(case)
    max = np.max(case)
    # axial
    fig, axes = plt.subplots(ncols=num_data, nrows=1, figsize=(num_data, 2))
    for i in range(num_data):
        axes[i].imshow(case[i, 3, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
        # axes[i].set_title('α=%d' % 0.1*i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "/interpolation.png")
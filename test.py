#!/usr/bin/env python

# Modified Horovod MNIST example

import os
import sys
import time

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import graphics
from utils import *
from tqdm import tqdm


learn = tf.contrib.learn

# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _print(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)

# need to modify(???)
def init_visualizations(hps, model, logdir):

    def sample_batch(y, eps):
        n_batch = hps.local_batch_train
        xs = []
        for i in range(int(np.ceil(len(eps) / n_batch))):
            xs.append(model.sample(
                y[i*n_batch:i*n_batch + n_batch], eps[i*n_batch:i*n_batch + n_batch]))
        return np.concatenate(xs)

    def draw_samples(epoch):
        if hvd.rank() != 0:
            return

        rows = 10 if hps.image_size <= 64 else 4
        cols = rows
        n_batch = rows*cols
        y = np.asarray([_y % hps.n_y for _y in (
            list(range(cols)) * rows)], dtype='int32')

        # temperatures = [0., .25, .5, .626, .75, .875, 1.] #previously
        temperatures = [0., .25, .5, .6, .7, .8, .9, 1.]

        x_samples = []
        x_samples.append(sample_batch(y, [.0]*n_batch))
        x_samples.append(sample_batch(y, [.25]*n_batch))
        x_samples.append(sample_batch(y, [.5]*n_batch))
        x_samples.append(sample_batch(y, [.6]*n_batch))
        x_samples.append(sample_batch(y, [.7]*n_batch))
        x_samples.append(sample_batch(y, [.8]*n_batch))
        x_samples.append(sample_batch(y, [.9]*n_batch))
        x_samples.append(sample_batch(y, [1.]*n_batch))
        # previously: 0, .25, .5, .625, .75, .875, 1.

        for i in range(len(x_samples)):
            x_sample = np.reshape(
                x_samples[i], (n_batch, hps.image_size, hps.image_size, 3))
            graphics.save_raster(x_sample, logdir +
                                 'epoch_{}_sample_{}.png'.format(epoch, i))

    return draw_samples



# ===
# Code for getting data
# ===
def get_data(hps, sess):
    hps.n_y = 1

    hps.local_batch_train = hps.n_batch_train
    # hps.local_batch_test = {64: 50, 32: 25, 16: 10, 8: 5, 4: 2, 2: 2, 1: 1}[
    #     hps.local_batch_train]  # round down to closest divisor of 50

    hps.local_batch_test = 1
    hps.local_batch_init = hps.n_batch_init

    print("Rank {} Batch sizes Train {} Test {} Init {}".format(
        hvd.rank(), hps.local_batch_train, hps.local_batch_test, hps.local_batch_init))

    hps.direct_iterator = True
    import get_data as v
    train_iterator, test_iterator, data_init = \
        v.get_data_for_test(sess, hps.data_dir, hvd.size(), hvd.rank(), hps.pmap, hps.fmap, hps.local_batch_train,
                   hps.local_batch_test, hps.local_batch_init)

    return train_iterator, test_iterator, data_init


def process_results(results):
    stats = ['loss', 'bits_x', 'bits_y', 'pred_loss']
    assert len(stats) == results.shape[0]
    res_dict = {}
    for i in range(len(stats)):
        res_dict[stats[i]] = "{:.4f}".format(results[i])
    return res_dict



def main(hps):

    # Initialize Horovod.
    hvd.init()

    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hvd.rank() + hvd.size() * hps.seed)
    np.random.seed(hvd.rank() + hvd.size() * hps.seed)

    # Get data and set train_its and valid_its
    train_iterator, test_iterator, data_init = get_data(hps, sess)
    hps.train_its, hps.test_its, hps.full_test_its = get_its(hps)

    # Create log dir
    logdir = os.path.abspath(hps.logdir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # Create model
    import model
    model = model.model(sess, hps, train_iterator, test_iterator, data_init)

    # Initialize visualization functions
    # visualise = init_visualizations(hps, model, logdir)

    infer(sess, model, hps, test_iterator)


def infer(sess, model, hps, iterator):
    # Example of using model in inference mode. Load saved model using hps.restore_path
    # Can provide x, y from files instead of dataset iterator
    # If model is uncondtional, always pass y = np.zeros([bs], dtype=np.int32)

    saver = tf.train.Saver()
    saver.restore(sess, hps.restore_path)

    if hps.direct_iterator:
        iterator = iterator.get_next()

    xs = []
    zs = []
    ori = []
    for it in tqdm(range(hps.full_test_its)):
        if hps.direct_iterator:
            # replace with x, y, attr if you're getting CelebA attributes, also modify get_data
            x, y = sess.run(iterator)
        else:
            x, y = iterator()

        ori.append(x)
        z = model.encode(x, y)
        x = model.decode(y, z)
        xs.append(x)
        zs.append(z)

    ori = np.concatenate(ori, axis=0)
    x = np.concatenate(xs, axis=0)
    z = np.concatenate(zs, axis=0)
    print(z.shape)

    # interpolation
    x1_2 = interpolation(y, z[1].reshape(1, 512), z[2].reshape(1, 512), model)
    x1_2 = np.reshape(x1_2, [11, 8, 8, 8])
    display_center_slices(x1_2, 8, 11, hps.logdir)
    np.save(hps.logdir+'/x.npy', x)
    np.save(hps.logdir+'/z.npy', z)
    ori = np.reshape(ori, [hps.full_test_its, 8, 8, 8])
    x = np.reshape(x, [hps.full_test_its, 8, 8, 8])
    z = np.reshape(z, [hps.full_test_its, 8, 8, 8])

    # calculate generalization
    gen = np.mean(np.reshape(abs(ori - x), [hps.full_test_its, 8*8*8]), axis=1)
    generalization = np.mean(gen)
    print('generalization = %f' % generalization)
    gen.reshape(-1,)
    np.savetxt(os.path.join(hps.logdir, 'generalization.csv'), gen, delimiter=",")

    ori_a = ori[:, 3, :]
    x_a = x[:, 3, :]
    visualize_slices(ori_a, x_a, hps.logdir)
    return zs


# Get number of training and validation iterations
def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())
    assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


'''
Create tensorflow session with horovod
'''
def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--restore_path", type=str, default='./model_best_loss.ckpt',
                        help="Location of checkpoint to restore")
    parser.add_argument("--inference", action="store_false",
                        help="Use in inference mode")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--data_dir", type=str, default='./data/1axis/tfrecord/',
                        help="Location of data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int,
                        default=6000, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=2000, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=1, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=2, help="Epochs between valid")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=8, help="Image size")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=1,
                        help="Coupling type: 0=additive, 1=affine")

    hps = parser.parse_args()  # So error if typo
    main(hps)

import os
import tensorflow as tf
import numpy as np
import glob

_FILES_SHUFFLE = 1024
_SHUFFLE_FACTOR = 4


def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'img_raw': tf.FixedLenFeature([8*8*8*1], tf.float32),
        'label': tf.FixedLenFeature([1], tf.int64),
        'shape': tf.FixedLenFeature([4], tf.int64)
    })
    data, label, shape = features['img_raw'], features['label'], features['shape']
    label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
    img = tf.reshape(data, [8, 8, 8, 1])
    return img, label


def input_fn(tfr_file, shards, rank, pmap, fmap, n_batch, is_training):
    files = tf.data.Dataset.list_files(tfr_file)
    if is_training:
        # each worker works on a subset of the data
        files = files.shard(shards, rank)
    if is_training:
        # shuffle order of files in shard
        files = files.shuffle(buffer_size=_FILES_SHUFFLE)
    dset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=fmap))
    if is_training:
        dset = dset.shuffle(buffer_size=n_batch * _SHUFFLE_FACTOR)
    dset = dset.repeat()
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=pmap)
    dset = dset.batch(n_batch)
    dset = dset.prefetch(1)
    itr = dset.make_one_shot_iterator()
    return itr


def get_tfr_file(data_dir, split):
    data_dir = os.path.join(data_dir, split)
    files = glob.glob(data_dir + '/*')
    return files


def get_data(sess, data_dir, shards, rank, pmap, fmap, n_batch_train, n_batch_test, n_batch_init):

    train_file = get_tfr_file(data_dir, 'train')
    valid_file = get_tfr_file(data_dir, 'validation')

    train_itr = input_fn(train_file, shards, rank, pmap,
                         fmap, n_batch_train, True)
    valid_itr = input_fn(valid_file, shards, rank, pmap,
                         fmap, n_batch_test, False)

    data_init = make_batch(sess, train_itr, n_batch_train, n_batch_init)

    return train_itr, valid_itr, data_init


def get_data_for_test(sess, data_dir, shards, rank, pmap, fmap, n_batch_train, n_batch_test, n_batch_init):

    train_file = get_tfr_file(data_dir, 'train')
    valid_file = get_tfr_file(data_dir, 'test')

    train_itr = input_fn(train_file, shards, rank, pmap,
                         fmap, n_batch_train, True)
    valid_itr = input_fn(valid_file, shards, rank, pmap,
                         fmap, n_batch_test, False)

    data_init = make_batch(sess, train_itr, n_batch_train, n_batch_init)

    return train_itr, valid_itr, data_init


def make_batch(sess, itr, itr_batch_size, required_batch_size):
    ib, rb = itr_batch_size, required_batch_size
    #assert rb % ib == 0
    k = int(np.ceil(rb / ib))
    xs, ys = [], []
    data = itr.get_next()
    for i in range(k):
        x, y = sess.run(data)
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs)[:rb], np.concatenate(ys)[:rb]
    return {'x': x, 'y': y}

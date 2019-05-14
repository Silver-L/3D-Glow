'''
@ TFRecord generator
@ Author: Zhihui Lu
@ Date: 2019/05/08
'''

import os
import tensorflow as tf
import numpy as np
import dataIO as io
from absl import flags, app

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings


# flag
FLAGS = flags.FLAGS
flags.DEFINE_string("data_list", "F:/data_info/glow/vessel/tfrecord_make.txt", "data list")
flags.DEFINE_string("outdir", "F:/data/tfrecord/vessel", "outdir")
flags.DEFINE_list("image_size", [8,8,8,1], "size of image")
flags.DEFINE_integer("num_per_tfrecord", 1000, "number per tfrecord")

def main(argv):
    # check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir)

    # load data set
    data_set = io.load_matrix_data(FLAGS.data_list, 'float32')

    # shuffle
    np.random.shuffle(data_set)

    num_per_tfrecord = int(FLAGS.num_per_tfrecord)
    num_of_total_image = data_set.shape[0]

    if (num_of_total_image % num_per_tfrecord != 0):
        num_of_recordfile = num_of_total_image // num_per_tfrecord + 1
    else:
        num_of_recordfile = num_of_total_image // num_per_tfrecord

    num_per_tfrecord_final = num_of_total_image - num_per_tfrecord * (num_of_recordfile - 1)

    print('number of total TFrecordfile: {}'.format(num_of_recordfile))

    # write TFrecord
    for i in range(num_of_recordfile):
        tfrecord_filename = os.path.join(FLAGS.outdir, 'recordfile_{}'.format(i + 1))
        # options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)  # compress
        write = tf.python_io.TFRecordWriter(tfrecord_filename)

        print('Writing recordfile_{}'.format(i+1))

        if i == num_of_recordfile - 1:
            loop_buf = num_per_tfrecord_final
        else :
            loop_buf = num_per_tfrecord

        for image_index in range(loop_buf):
            image = data_set[image_index + i*num_per_tfrecord].flatten()

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=image)),
                    'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                    'shape':tf.train.Feature(int64_list=tf.train.Int64List(value=FLAGS.image_size))
                }))

            write.write(example.SerializeToString())
        write.close()


if __name__ == '__main__':
    app.run(main)

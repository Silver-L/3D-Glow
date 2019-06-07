import os
import tensorflow as tf
import pandas as pd
from absl import flags, app

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings

# flag
FLAGS = flags.FLAGS
flags.DEFINE_string("dir", "C:/Users/LUZHIHUI/Desktop/tensorboard", "log folder")


def main(argv):

    # turn off log message
    tf.logging.set_verbosity(tf.logging.FATAL)

    # check folder
    if not os.path.exists(os.path.join(FLAGS.dir, 'tensorboard')):
        os.makedirs(os.path.join(FLAGS.dir, 'tensorboard'))


    value_loss = tf.Variable(0.0)
    tf.summary.scalar("train_loss", value_loss)
    merge_op = tf.summary.merge_all()

    # load data
    train_txt = pd.read_csv(os.path.join(FLAGS.dir, 'train.txt'), skiprows=[0],
                            sep=',|:|"', engine='python', header=None)
    val_txt = pd.read_csv(os.path.join(FLAGS.dir, 'test.txt'), skiprows=[0],
                            sep=',|:|"', engine='python', header=None)

    train_data = train_txt.values
    val_data = val_txt.values

    # initializer
    init_op = tf.initialize_all_variables()

    with tf.Session(config = config(index="0")) as sess:
        # prepare tensorboard
        writer_train = tf.summary.FileWriter(os.path.join(FLAGS.dir, 'tensorboard', 'train'))
        writer_val = tf.summary.FileWriter(os.path.join(FLAGS.dir, 'tensorboard', 'val'))

        value_loss = tf.Variable(0.0)
        tf.summary.scalar("train_loss", value_loss)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        for i in range(train_data.shape[0]):
            summary_train_loss = sess.run(merge_op, {value_loss: train_data[i][20]})
            writer_train.add_summary(summary_train_loss, train_data[i][3])

        for i in range(val_data.shape[0]):
            summary_val_loss = sess.run(merge_op, {value_loss: val_data[i][16]})
            writer_val.add_summary(summary_val_loss, val_data[i][3])

# session config
def config(index = "0"):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=index , # specify GPU number
            allow_growth=True
        )
    )
    return config


if __name__ == '__main__':
    app.run(main)

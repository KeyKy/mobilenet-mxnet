import argparse
from common import modelzoo, find_mxnet
import mxnet as mx
import time
import os,sys
import logging
import numpy as np

def score(prefix, epoch, data_val, metrics, gpus, batch_size, rgb_mean=None, mean_img=None,
          image_shape='3,224,224', data_nthreads=4, label_name='softmax_label', max_num_examples=None):
    # create data iterator
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    if mean_img is not None:
        mean_args = {'mean_img':mean_img}
    elif rgb_mean is not None:
        rgb_mean = [float(i) for i in rgb_mean.split(',')]
        mean_args = {'mean_r':rgb_mean[0], 'mean_g':rgb_mean[1],
          'mean_b':rgb_mean[2]}

    data = mx.io.ImageRecordIter(
        path_imgrec        = data_val,
        label_width        = 1,
        preprocess_threads = data_nthreads,
        batch_size         = batch_size,
        data_shape         = data_shape,
        label_name         = label_name,
        rand_crop          = False,
        rand_mirror        = False,
        resize=256,
        scale=0.017,
        **mean_args)
    if isinstance(prefix, str):
        # download model
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # create module
    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]

    mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name,])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)
    if not isinstance(metrics, list):
        metrics = [metrics,]
    tic = time.time()
    num = 0
    for batch in data:
        mod.forward(batch, is_train=False)

        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if max_num_examples is not None and num > max_num_examples:
            break
    return (num / (time.time() - tic), )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--prefix', type=str, required=True,
                        help = 'the model prefix name.')
    parser.add_argument('--epoch', type=int, required=True, help='epoch load')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--rgb-mean', type=str, default='0,0,0')
    parser.add_argument('--data-val', type=str, required=True)
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=4,
                        help='number of threads for data decoding')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k = 5)]

    (speed,) = score(metrics = metrics, **vars(args))
    logging.info('Finished with %f images per second', speed)

    for m in metrics:
        logging.info(m.get())

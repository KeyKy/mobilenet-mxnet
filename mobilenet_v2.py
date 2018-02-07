import mxnet as mx
def get_symbol(num_classes, **kwargs):
    if 'use_global_stats' not in kwargs:
        use_global_stats = False
    else:
        use_global_stats = kwargs['use_global_stats']
    data = mx.symbol.Variable(name='data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1 , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv1_scale = conv1_bn
    relu1 = mx.symbol.Activation(name='relu1', data=conv1_scale , act_type='relu')
    conv2_1_expand = mx.symbol.Convolution(name='conv2_1_expand', data=relu1 , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv2_1_expand_bn = mx.symbol.BatchNorm(name='conv2_1_expand_bn', data=conv2_1_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_1_expand_scale = conv2_1_expand_bn
    relu2_1_expand = mx.symbol.Activation(name='relu2_1_expand', data=conv2_1_expand_scale , act_type='relu')
    conv2_1_dwise = mx.symbol.Convolution(name='conv2_1_dwise', data=relu2_1_expand , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=32)
    conv2_1_dwise_bn = mx.symbol.BatchNorm(name='conv2_1_dwise_bn', data=conv2_1_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_1_dwise_scale = conv2_1_dwise_bn
    relu2_1_dwise = mx.symbol.Activation(name='relu2_1_dwise', data=conv2_1_dwise_scale , act_type='relu')
    conv2_1_linear = mx.symbol.Convolution(name='conv2_1_linear', data=relu2_1_dwise , num_filter=16, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv2_1_linear_bn = mx.symbol.BatchNorm(name='conv2_1_linear_bn', data=conv2_1_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_1_linear_scale = conv2_1_linear_bn
    conv2_2_expand = mx.symbol.Convolution(name='conv2_2_expand', data=conv2_1_linear_scale , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv2_2_expand_bn = mx.symbol.BatchNorm(name='conv2_2_expand_bn', data=conv2_2_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_2_expand_scale = conv2_2_expand_bn
    relu2_2_expand = mx.symbol.Activation(name='relu2_2_expand', data=conv2_2_expand_scale , act_type='relu')
    conv2_2_dwise = mx.symbol.Convolution(name='conv2_2_dwise', data=relu2_2_expand , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=96)
    conv2_2_dwise_bn = mx.symbol.BatchNorm(name='conv2_2_dwise_bn', data=conv2_2_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_2_dwise_scale = conv2_2_dwise_bn
    relu2_2_dwise = mx.symbol.Activation(name='relu2_2_dwise', data=conv2_2_dwise_scale , act_type='relu')
    conv2_2_linear = mx.symbol.Convolution(name='conv2_2_linear', data=relu2_2_dwise , num_filter=24, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv2_2_linear_bn = mx.symbol.BatchNorm(name='conv2_2_linear_bn', data=conv2_2_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_2_linear_scale = conv2_2_linear_bn
    conv3_1_expand = mx.symbol.Convolution(name='conv3_1_expand', data=conv2_2_linear_scale , num_filter=144, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv3_1_expand_bn = mx.symbol.BatchNorm(name='conv3_1_expand_bn', data=conv3_1_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_1_expand_scale = conv3_1_expand_bn
    relu3_1_expand = mx.symbol.Activation(name='relu3_1_expand', data=conv3_1_expand_scale , act_type='relu')
    conv3_1_dwise = mx.symbol.Convolution(name='conv3_1_dwise', data=relu3_1_expand , num_filter=144, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=144)
    conv3_1_dwise_bn = mx.symbol.BatchNorm(name='conv3_1_dwise_bn', data=conv3_1_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_1_dwise_scale = conv3_1_dwise_bn
    relu3_1_dwise = mx.symbol.Activation(name='relu3_1_dwise', data=conv3_1_dwise_scale , act_type='relu')
    conv3_1_linear = mx.symbol.Convolution(name='conv3_1_linear', data=relu3_1_dwise , num_filter=24, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv3_1_linear_bn = mx.symbol.BatchNorm(name='conv3_1_linear_bn', data=conv3_1_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_1_linear_scale = conv3_1_linear_bn
    block_3_1 = mx.symbol.broadcast_add(name='block_3_1', *[conv2_2_linear_scale,conv3_1_linear_scale] )
    conv3_2_expand = mx.symbol.Convolution(name='conv3_2_expand', data=block_3_1 , num_filter=144, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv3_2_expand_bn = mx.symbol.BatchNorm(name='conv3_2_expand_bn', data=conv3_2_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_2_expand_scale = conv3_2_expand_bn
    relu3_2_expand = mx.symbol.Activation(name='relu3_2_expand', data=conv3_2_expand_scale , act_type='relu')
    conv3_2_dwise = mx.symbol.Convolution(name='conv3_2_dwise', data=relu3_2_expand , num_filter=144, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=144)
    conv3_2_dwise_bn = mx.symbol.BatchNorm(name='conv3_2_dwise_bn', data=conv3_2_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_2_dwise_scale = conv3_2_dwise_bn
    relu3_2_dwise = mx.symbol.Activation(name='relu3_2_dwise', data=conv3_2_dwise_scale , act_type='relu')
    conv3_2_linear = mx.symbol.Convolution(name='conv3_2_linear', data=relu3_2_dwise , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv3_2_linear_bn = mx.symbol.BatchNorm(name='conv3_2_linear_bn', data=conv3_2_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_2_linear_scale = conv3_2_linear_bn
    conv4_1_expand = mx.symbol.Convolution(name='conv4_1_expand', data=conv3_2_linear_scale , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_1_expand_bn = mx.symbol.BatchNorm(name='conv4_1_expand_bn', data=conv4_1_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_1_expand_scale = conv4_1_expand_bn
    relu4_1_expand = mx.symbol.Activation(name='relu4_1_expand', data=conv4_1_expand_scale , act_type='relu')
    conv4_1_dwise = mx.symbol.Convolution(name='conv4_1_dwise', data=relu4_1_expand , num_filter=192, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=192)
    conv4_1_dwise_bn = mx.symbol.BatchNorm(name='conv4_1_dwise_bn', data=conv4_1_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_1_dwise_scale = conv4_1_dwise_bn
    relu4_1_dwise = mx.symbol.Activation(name='relu4_1_dwise', data=conv4_1_dwise_scale , act_type='relu')
    conv4_1_linear = mx.symbol.Convolution(name='conv4_1_linear', data=relu4_1_dwise , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_1_linear_bn = mx.symbol.BatchNorm(name='conv4_1_linear_bn', data=conv4_1_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_1_linear_scale = conv4_1_linear_bn
    block_4_1 = mx.symbol.broadcast_add(name='block_4_1', *[conv3_2_linear_scale,conv4_1_linear_scale] )
    conv4_2_expand = mx.symbol.Convolution(name='conv4_2_expand', data=block_4_1 , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_2_expand_bn = mx.symbol.BatchNorm(name='conv4_2_expand_bn', data=conv4_2_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_2_expand_scale = conv4_2_expand_bn
    relu4_2_expand = mx.symbol.Activation(name='relu4_2_expand', data=conv4_2_expand_scale , act_type='relu')
    conv4_2_dwise = mx.symbol.Convolution(name='conv4_2_dwise', data=relu4_2_expand , num_filter=192, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=192)
    conv4_2_dwise_bn = mx.symbol.BatchNorm(name='conv4_2_dwise_bn', data=conv4_2_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_2_dwise_scale = conv4_2_dwise_bn
    relu4_2_dwise = mx.symbol.Activation(name='relu4_2_dwise', data=conv4_2_dwise_scale , act_type='relu')
    conv4_2_linear = mx.symbol.Convolution(name='conv4_2_linear', data=relu4_2_dwise , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_2_linear_bn = mx.symbol.BatchNorm(name='conv4_2_linear_bn', data=conv4_2_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_2_linear_scale = conv4_2_linear_bn
    block_4_2 = mx.symbol.broadcast_add(name='block_4_2', *[block_4_1,conv4_2_linear_scale] )
    conv4_3_expand = mx.symbol.Convolution(name='conv4_3_expand', data=block_4_2 , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_3_expand_bn = mx.symbol.BatchNorm(name='conv4_3_expand_bn', data=conv4_3_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_3_expand_scale = conv4_3_expand_bn
    relu4_3_expand = mx.symbol.Activation(name='relu4_3_expand', data=conv4_3_expand_scale , act_type='relu')
    conv4_3_dwise = mx.symbol.Convolution(name='conv4_3_dwise', data=relu4_3_expand , num_filter=192, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=192)
    conv4_3_dwise_bn = mx.symbol.BatchNorm(name='conv4_3_dwise_bn', data=conv4_3_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_3_dwise_scale = conv4_3_dwise_bn
    relu4_3_dwise = mx.symbol.Activation(name='relu4_3_dwise', data=conv4_3_dwise_scale , act_type='relu')
    conv4_3_linear = mx.symbol.Convolution(name='conv4_3_linear', data=relu4_3_dwise , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_3_linear_bn = mx.symbol.BatchNorm(name='conv4_3_linear_bn', data=conv4_3_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_3_linear_scale = conv4_3_linear_bn
    conv4_4_expand = mx.symbol.Convolution(name='conv4_4_expand', data=conv4_3_linear_scale , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_4_expand_bn = mx.symbol.BatchNorm(name='conv4_4_expand_bn', data=conv4_4_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_4_expand_scale = conv4_4_expand_bn
    relu4_4_expand = mx.symbol.Activation(name='relu4_4_expand', data=conv4_4_expand_scale , act_type='relu')
    conv4_4_dwise = mx.symbol.Convolution(name='conv4_4_dwise', data=relu4_4_expand , num_filter=384, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=384)
    conv4_4_dwise_bn = mx.symbol.BatchNorm(name='conv4_4_dwise_bn', data=conv4_4_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_4_dwise_scale = conv4_4_dwise_bn
    relu4_4_dwise = mx.symbol.Activation(name='relu4_4_dwise', data=conv4_4_dwise_scale , act_type='relu')
    conv4_4_linear = mx.symbol.Convolution(name='conv4_4_linear', data=relu4_4_dwise , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_4_linear_bn = mx.symbol.BatchNorm(name='conv4_4_linear_bn', data=conv4_4_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_4_linear_scale = conv4_4_linear_bn
    block_4_4 = mx.symbol.broadcast_add(name='block_4_4', *[conv4_3_linear_scale,conv4_4_linear_scale] )
    conv4_5_expand = mx.symbol.Convolution(name='conv4_5_expand', data=block_4_4 , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_5_expand_bn = mx.symbol.BatchNorm(name='conv4_5_expand_bn', data=conv4_5_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_5_expand_scale = conv4_5_expand_bn
    relu4_5_expand = mx.symbol.Activation(name='relu4_5_expand', data=conv4_5_expand_scale , act_type='relu')
    conv4_5_dwise = mx.symbol.Convolution(name='conv4_5_dwise', data=relu4_5_expand , num_filter=384, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=384)
    conv4_5_dwise_bn = mx.symbol.BatchNorm(name='conv4_5_dwise_bn', data=conv4_5_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_5_dwise_scale = conv4_5_dwise_bn
    relu4_5_dwise = mx.symbol.Activation(name='relu4_5_dwise', data=conv4_5_dwise_scale , act_type='relu')
    conv4_5_linear = mx.symbol.Convolution(name='conv4_5_linear', data=relu4_5_dwise , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_5_linear_bn = mx.symbol.BatchNorm(name='conv4_5_linear_bn', data=conv4_5_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_5_linear_scale = conv4_5_linear_bn
    block_4_5 = mx.symbol.broadcast_add(name='block_4_5', *[block_4_4,conv4_5_linear_scale] )
    conv4_6_expand = mx.symbol.Convolution(name='conv4_6_expand', data=block_4_5 , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_6_expand_bn = mx.symbol.BatchNorm(name='conv4_6_expand_bn', data=conv4_6_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_6_expand_scale = conv4_6_expand_bn
    relu4_6_expand = mx.symbol.Activation(name='relu4_6_expand', data=conv4_6_expand_scale , act_type='relu')
    conv4_6_dwise = mx.symbol.Convolution(name='conv4_6_dwise', data=relu4_6_expand , num_filter=384, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=384)
    conv4_6_dwise_bn = mx.symbol.BatchNorm(name='conv4_6_dwise_bn', data=conv4_6_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_6_dwise_scale = conv4_6_dwise_bn
    relu4_6_dwise = mx.symbol.Activation(name='relu4_6_dwise', data=conv4_6_dwise_scale , act_type='relu')
    conv4_6_linear = mx.symbol.Convolution(name='conv4_6_linear', data=relu4_6_dwise , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_6_linear_bn = mx.symbol.BatchNorm(name='conv4_6_linear_bn', data=conv4_6_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_6_linear_scale = conv4_6_linear_bn
    block_4_6 = mx.symbol.broadcast_add(name='block_4_6', *[block_4_5,conv4_6_linear_scale] )
    conv4_7_expand = mx.symbol.Convolution(name='conv4_7_expand', data=block_4_6 , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_7_expand_bn = mx.symbol.BatchNorm(name='conv4_7_expand_bn', data=conv4_7_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_7_expand_scale = conv4_7_expand_bn
    relu4_7_expand = mx.symbol.Activation(name='relu4_7_expand', data=conv4_7_expand_scale , act_type='relu')
    conv4_7_dwise = mx.symbol.Convolution(name='conv4_7_dwise', data=relu4_7_expand , num_filter=384, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=384)
    conv4_7_dwise_bn = mx.symbol.BatchNorm(name='conv4_7_dwise_bn', data=conv4_7_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_7_dwise_scale = conv4_7_dwise_bn
    relu4_7_dwise = mx.symbol.Activation(name='relu4_7_dwise', data=conv4_7_dwise_scale , act_type='relu')
    conv4_7_linear = mx.symbol.Convolution(name='conv4_7_linear', data=relu4_7_dwise , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_7_linear_bn = mx.symbol.BatchNorm(name='conv4_7_linear_bn', data=conv4_7_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_7_linear_scale = conv4_7_linear_bn
    conv5_1_expand = mx.symbol.Convolution(name='conv5_1_expand', data=conv4_7_linear_scale , num_filter=576, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_1_expand_bn = mx.symbol.BatchNorm(name='conv5_1_expand_bn', data=conv5_1_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_1_expand_scale = conv5_1_expand_bn
    relu5_1_expand = mx.symbol.Activation(name='relu5_1_expand', data=conv5_1_expand_scale , act_type='relu')
    conv5_1_dwise = mx.symbol.Convolution(name='conv5_1_dwise', data=relu5_1_expand , num_filter=576, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=576)
    conv5_1_dwise_bn = mx.symbol.BatchNorm(name='conv5_1_dwise_bn', data=conv5_1_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_1_dwise_scale = conv5_1_dwise_bn
    relu5_1_dwise = mx.symbol.Activation(name='relu5_1_dwise', data=conv5_1_dwise_scale , act_type='relu')
    conv5_1_linear = mx.symbol.Convolution(name='conv5_1_linear', data=relu5_1_dwise , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_1_linear_bn = mx.symbol.BatchNorm(name='conv5_1_linear_bn', data=conv5_1_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_1_linear_scale = conv5_1_linear_bn
    block_5_1 = mx.symbol.broadcast_add(name='block_5_1', *[conv4_7_linear_scale,conv5_1_linear_scale] )
    conv5_2_expand = mx.symbol.Convolution(name='conv5_2_expand', data=block_5_1 , num_filter=576, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_2_expand_bn = mx.symbol.BatchNorm(name='conv5_2_expand_bn', data=conv5_2_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_2_expand_scale = conv5_2_expand_bn
    relu5_2_expand = mx.symbol.Activation(name='relu5_2_expand', data=conv5_2_expand_scale , act_type='relu')
    conv5_2_dwise = mx.symbol.Convolution(name='conv5_2_dwise', data=relu5_2_expand , num_filter=576, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=576)
    conv5_2_dwise_bn = mx.symbol.BatchNorm(name='conv5_2_dwise_bn', data=conv5_2_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_2_dwise_scale = conv5_2_dwise_bn
    relu5_2_dwise = mx.symbol.Activation(name='relu5_2_dwise', data=conv5_2_dwise_scale , act_type='relu')
    conv5_2_linear = mx.symbol.Convolution(name='conv5_2_linear', data=relu5_2_dwise , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_2_linear_bn = mx.symbol.BatchNorm(name='conv5_2_linear_bn', data=conv5_2_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_2_linear_scale = conv5_2_linear_bn
    block_5_2 = mx.symbol.broadcast_add(name='block_5_2', *[block_5_1,conv5_2_linear_scale] )
    conv5_3_expand = mx.symbol.Convolution(name='conv5_3_expand', data=block_5_2 , num_filter=576, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_3_expand_bn = mx.symbol.BatchNorm(name='conv5_3_expand_bn', data=conv5_3_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_3_expand_scale = conv5_3_expand_bn
    relu5_3_expand = mx.symbol.Activation(name='relu5_3_expand', data=conv5_3_expand_scale , act_type='relu')
    conv5_3_dwise = mx.symbol.Convolution(name='conv5_3_dwise', data=relu5_3_expand , num_filter=576, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=576)
    conv5_3_dwise_bn = mx.symbol.BatchNorm(name='conv5_3_dwise_bn', data=conv5_3_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_3_dwise_scale = conv5_3_dwise_bn
    relu5_3_dwise = mx.symbol.Activation(name='relu5_3_dwise', data=conv5_3_dwise_scale , act_type='relu')
    conv5_3_linear = mx.symbol.Convolution(name='conv5_3_linear', data=relu5_3_dwise , num_filter=160, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_3_linear_bn = mx.symbol.BatchNorm(name='conv5_3_linear_bn', data=conv5_3_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_3_linear_scale = conv5_3_linear_bn
    conv6_1_expand = mx.symbol.Convolution(name='conv6_1_expand', data=conv5_3_linear_scale , num_filter=960, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv6_1_expand_bn = mx.symbol.BatchNorm(name='conv6_1_expand_bn', data=conv6_1_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_1_expand_scale = conv6_1_expand_bn
    relu6_1_expand = mx.symbol.Activation(name='relu6_1_expand', data=conv6_1_expand_scale , act_type='relu')
    conv6_1_dwise = mx.symbol.Convolution(name='conv6_1_dwise', data=relu6_1_expand , num_filter=960, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=960)
    conv6_1_dwise_bn = mx.symbol.BatchNorm(name='conv6_1_dwise_bn', data=conv6_1_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_1_dwise_scale = conv6_1_dwise_bn
    relu6_1_dwise = mx.symbol.Activation(name='relu6_1_dwise', data=conv6_1_dwise_scale , act_type='relu')
    conv6_1_linear = mx.symbol.Convolution(name='conv6_1_linear', data=relu6_1_dwise , num_filter=160, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv6_1_linear_bn = mx.symbol.BatchNorm(name='conv6_1_linear_bn', data=conv6_1_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_1_linear_scale = conv6_1_linear_bn
    block_6_1 = mx.symbol.broadcast_add(name='block_6_1', *[conv5_3_linear_scale,conv6_1_linear_scale] )
    conv6_2_expand = mx.symbol.Convolution(name='conv6_2_expand', data=block_6_1 , num_filter=960, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv6_2_expand_bn = mx.symbol.BatchNorm(name='conv6_2_expand_bn', data=conv6_2_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_2_expand_scale = conv6_2_expand_bn
    relu6_2_expand = mx.symbol.Activation(name='relu6_2_expand', data=conv6_2_expand_scale , act_type='relu')
    conv6_2_dwise = mx.symbol.Convolution(name='conv6_2_dwise', data=relu6_2_expand , num_filter=960, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=960)
    conv6_2_dwise_bn = mx.symbol.BatchNorm(name='conv6_2_dwise_bn', data=conv6_2_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_2_dwise_scale = conv6_2_dwise_bn
    relu6_2_dwise = mx.symbol.Activation(name='relu6_2_dwise', data=conv6_2_dwise_scale , act_type='relu')
    conv6_2_linear = mx.symbol.Convolution(name='conv6_2_linear', data=relu6_2_dwise , num_filter=160, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv6_2_linear_bn = mx.symbol.BatchNorm(name='conv6_2_linear_bn', data=conv6_2_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_2_linear_scale = conv6_2_linear_bn
    block_6_2 = mx.symbol.broadcast_add(name='block_6_2', *[block_6_1,conv6_2_linear_scale] )
    conv6_3_expand = mx.symbol.Convolution(name='conv6_3_expand', data=block_6_2 , num_filter=960, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv6_3_expand_bn = mx.symbol.BatchNorm(name='conv6_3_expand_bn', data=conv6_3_expand , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_3_expand_scale = conv6_3_expand_bn
    relu6_3_expand = mx.symbol.Activation(name='relu6_3_expand', data=conv6_3_expand_scale , act_type='relu')
    conv6_3_dwise = mx.symbol.Convolution(name='conv6_3_dwise', data=relu6_3_expand , num_filter=960, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=960)
    conv6_3_dwise_bn = mx.symbol.BatchNorm(name='conv6_3_dwise_bn', data=conv6_3_dwise , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_3_dwise_scale = conv6_3_dwise_bn
    relu6_3_dwise = mx.symbol.Activation(name='relu6_3_dwise', data=conv6_3_dwise_scale , act_type='relu')
    conv6_3_linear = mx.symbol.Convolution(name='conv6_3_linear', data=relu6_3_dwise , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv6_3_linear_bn = mx.symbol.BatchNorm(name='conv6_3_linear_bn', data=conv6_3_linear , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_3_linear_scale = conv6_3_linear_bn
    conv6_4 = mx.symbol.Convolution(name='conv6_4', data=conv6_3_linear_scale , num_filter=1280, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv6_4_bn = mx.symbol.BatchNorm(name='conv6_4_bn', data=conv6_4 , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_4_scale = conv6_4_bn
    relu6_4 = mx.symbol.Activation(name='relu6_4', data=conv6_4_scale , act_type='relu')
    pool6 = mx.symbol.Pooling(name='pool6', data=relu6_4 , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
    fc7 = mx.symbol.Convolution(name='fc7', data=pool6 , num_filter=1000, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
    flatten = mx.sym.flatten(fc7, name='fc7')
    softmax = mx.symbol.SoftmaxOutput(name='prob', data=flatten )
    return softmax



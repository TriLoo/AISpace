# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '.')

import mxnet as mx
# from mxnet import nd
ctx = mx.gpu(4)


def test_1():
    a = mx.nd.array([3, 2, 3, 4, 5])
    a = a.expand_dims(axis=0)
    b = mx.nd.tile(a, reps=(11, 1))
    b = b.expand_dims(axis=0).expand_dims(axis=0)
    b = mx.nd.tile(b, reps=(2, 2, 1, 1))

    c = mx.nd.RightLeftPooling(b)
    print('in: ', b)
    print('out: ', c)
    print('shape of c: ', c)


def test_2():
    a = mx.nd.array([3, 2, 3, 4, 5])
    a = a.expand_dims(axis=0)
    b = mx.nd.tile(a, reps=(5, 1))
    b = b.expand_dims(axis=0).expand_dims(axis=0)
    # b = mx.nd.tile(b, reps=(1, 2, 1, 1))

    data = mx.sym.Variable('data')
    out = mx.sym.RightLeftPooling(data)
    print('out: ', out)
    arr_grad = mx.nd.empty((1, 1, 5, 5))
    exec1 = out.bind(mx.cpu(), args={"data":b}, args_grad=[arr_grad])
    exec1.forward(is_train=True)
    out1 = exec1.outputs[0]
    grad_arr = mx.nd.ones((1, 1, 5, 5))
    exec1.backward([grad_arr])

    c = mx.nd.RightLeftPooling(b)

    print('in: ', b)
    print('out: ', c)
    print('shape of c: ', c)
    print('sym out: ', out1)
    print('shape of sym out: ', out1.shape)
    print('grad out: ', arr_grad)


def test_gpu_1():
    a = mx.nd.array([3, 2, 3, 4, 5], ctx)
    a = a.expand_dims(axis=0)
    b = mx.nd.tile(a, reps=(11, 1))
    b = b.expand_dims(axis=0).expand_dims(axis=0)
    b = mx.nd.tile(b, reps=(2, 2, 1, 1))

    c = mx.nd.RightLeftPooling(b)
    # c = mx.nd.contrib.RightLeftPooling(b)
    print('in: ', b)
    print('out: ', c)
    print('shape of c: ', c)


def test_gpu_2():
    a = mx.nd.array([3, 2, 3, 4, 5], ctx)
    a = a.expand_dims(axis=0)
    b = mx.nd.tile(a, reps=(5, 1))
    b = b.expand_dims(axis=0).expand_dims(axis=0)
    # b = mx.nd.tile(b, reps=(1, 2, 1, 1))

    data = mx.sym.Variable('data')
    out = mx.sym.RightLeftPooling(data)
    print('out: ', out)
    arr_grad = mx.nd.empty((1, 1, 5, 5), ctx)
    exec1 = out.bind(ctx, args={"data":b}, args_grad=[arr_grad])
    exec1.forward(is_train=True)
    out1 = exec1.outputs[0]
    grad_arr = mx.nd.ones((1, 1, 5, 5), ctx)
    exec1.backward([grad_arr])

    c = mx.nd.RightLeftPooling(b)

    print('in: ', b)
    print('out: ', c)
    print('shape of c: ', c)
    print('sym out: ', out1)
    print('shape of sym out: ', out1.shape)
    print('grad out: ', arr_grad)
    

if __name__ == '__main__':
    test_1()
    # test_2()
    # test_gpu_1()
    # test_gpu_2()


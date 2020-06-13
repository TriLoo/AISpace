# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '.')

import mxnet as mx
# from mxnet import nd


def test_1():
    a = mx.nd.array([5, 4, 3, 2, 3])
    a = a.expand_dims(axis=1)
    b = mx.nd.tile(a, reps=(1, 11))
    b = b.expand_dims(axis=0).expand_dims(axis=0)
    b = mx.nd.tile(b, reps=(2, 2, 1, 1))

    c = mx.nd.UpBottomPooling(b)
    print('in: ', b)
    print('out: ', c)
    print('shape of c: ', c)


def test_2():
    a = mx.nd.array([5, 4, 3, 2, 3])
    a = a.expand_dims(axis=1)
    b = mx.nd.tile(a, reps=(1, 5))
    b = b.expand_dims(axis=0).expand_dims(axis=0)
    # b = mx.nd.tile(b, reps=(1, 2, 1, 1))

    data = mx.sym.Variable('data')
    out = mx.sym.UpBottomPooling(data)
    print('out: ', out)
    arr_grad = mx.nd.empty((1, 1, 5, 5))
    exec1 = out.bind(mx.cpu(), args={"data":b}, args_grad=[arr_grad])
    exec1.forward(is_train=True)
    out1 = exec1.outputs[0]
    grad_arr = mx.nd.ones((1, 1, 5, 5))
    exec1.backward([grad_arr])

    c = mx.nd.UpBottomPooling(b)

    print('in: ', b)
    print('out: ', c)
    print('shape of c: ', c.shape)
    print('sym out: ', out1)
    print('shape of sym out: ', out1.shape)
    print('grad out: ', arr_grad)


def test_gpu_1():
    a = mx.nd.array([5, 4, 3, 2, 3], mx.gpu(1))
    a = a.expand_dims(axis=1)
    b = mx.nd.tile(a, reps=(1, 11))
    b = b.expand_dims(axis=0).expand_dims(axis=0)
    b = mx.nd.tile(b, reps=(2, 2, 1, 1))

    c = mx.nd.UpBottomPooling(b)
    # c = mx.nd.contrib.UpBottomPooling(b)
    print('in: ', b)
    print('out: ', c)
    print('shape of c: ', c.shape)


def test_gpu_2():
    a = mx.nd.array([5, 4, 3, 2, 3], mx.gpu(1))
    a = a.expand_dims(axis=1)
    b = mx.nd.tile(a, reps=(1, 5))
    b = b.expand_dims(axis=0).expand_dims(axis=0)
    # b = mx.nd.tile(b, reps=(1, 2, 1, 1))

    data = mx.sym.Variable('data')
    out = mx.sym.UpBottomPooling(data)
    print('out: ', out)
    arr_grad = mx.nd.empty((1, 1, 5, 5), mx.gpu(1))
    exec1 = out.bind(mx.gpu(1), args={"data":b}, args_grad=[arr_grad])
    exec1.forward(is_train=True)
    out1 = exec1.outputs[0]
    grad_arr = mx.nd.ones((1, 1, 5, 5), mx.gpu(1))
    exec1.backward([grad_arr])

    c = mx.nd.UpBottomPooling(b)

    print('in: ', b)
    print('out: ', c)
    print('shape of c: ', c.shape)
    print('sym out: ', out1)
    print('shape of sym out: ', out1.shape)
    print('grad out: ', arr_grad)
    

if __name__ == '__main__':
    # a = mx.nd.array([1, 2, 3, 4, 5])
    # test_1()
    # test_2()
    test_gpu_1()
    # test_gpu_2()


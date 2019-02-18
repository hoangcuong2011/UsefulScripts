Learning rate - batch size is always daunting.

The default formula: Adam with learning rate 0.0001 and batch size: 32
This gives a decent result, but not sure the best (e.g. 0.6286 accuracy)
More importantly, at some point the loss function can increase. It is very painful.

So technically, we should train a model with decay learning rate.
I found a good strategy is using stepwise with decay rate: 0.75

Meanwhile, sometimes we might want to increase the batch size.

With only 1 single GPU, this might not help be that helpful.

But training a model with multiple GPUs, this can be super-helpful.

A common standard is the learning rate should be scaled with batch size. (e.g. larger batch size -> larger learning rate)

In the end, I provide an example I use for training neural network with batch size 2048 and 32, with default learning rate: 0.16 and decay rate 0.75



    Column A: epoch
    Column B: np.float(epoch/2)
    Column C: default learning rate * (0.75 ** np.float(epoch/2))
    with default learning rate for batchsize 2048: 0.16

    Column D: default learning rate * (0.75 ** np.float(epoch/2))
    with default learning rate for batchsize 32: 0.16/2048*32 = 0.0025
    
The question is: Is it really true that learning rate should be scaled with batch size???


------

    A	B	batch size 2048	equivalent batch size 32
    1	0	0.16	0.0025
    2	1	0.12	0.001875
    3	1	0.12	0.001875
    4	2	0.09	0.00140625
    5	2	0.09	0.00140625
    6	3	0.0675	0.001054688
    7	3	0.0675	0.001054688
    8	4	0.050625	0.000791016
    9	4	0.050625	0.000791016
    10	5	0.03796875	0.000593262
    11	5	0.03796875	0.000593262
    12	6	0.028476563	0.000444946
    13	6	0.028476563	0.000444946
    14	7	0.021357422	0.00033371
    15	7	0.021357422	0.00033371
    16	8	0.016018066	0.000250282
    17	8	0.016018066	0.000250282
    18	9	0.01201355	0.000187712
    19	9	0.01201355	0.000187712
    20	10	0.009010162	0.000140784

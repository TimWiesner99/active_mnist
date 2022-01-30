# active_mnist
Small program that let's you sample from the MNIST handwritten digits dataset, but with random rotation, scaling and translation transformations applied.

The class can (for example) be used in the following way:
```
import torch
from active_mnist import Active_MNIST, plot_example

dtype = torch.FloatTensor

dataloader = Active_MNIST(image_path='../data/train-images.idx3-ubyte',
                        label_path='../data/train-labels.idx1-ubyte',
                        batch_size=1024, max_iterations=64,
                        dtype=dtype)

or_img, trns_img, lbl, pose = dataloader.sample(100)
plot_example(or_img, trns_img, lbl, pose, number_examples=3)
```
![index](https://user-images.githubusercontent.com/62284314/142396995-510d546e-ae6c-4e26-a382-49d8132118ae.jpg)

The dataset is loaded when the class is initialized with `batch_size * max_iterations` samples.

The `sample()` function returns the original images, the transformed images, the labels and a vector containing the transformations. The transformations vector contains five numbers, which are: x-scaling, y-scaling, rotation, x-translation, y-translation.

The `plot_example()` function takes the four arrays returned by the `sample()` function as input and plots one example by default. A larger number of examples can also be printed, as specified by the `number_examples` argument. 

It is also possible to use the class as an iterator in a machine learning training loop, as shown below:

```
dataloader.load_data(perc_normal=0.1, perc_distractors=0.1)

for or_img, trns_img, cls, pose in dataloader:
  # training epoch
```
The data loader will then return `max_iterations` batches of size `batch_size`. The length of an epoch is thus specified in the initialization of the `Active_MNIST` class. 

The `load_data()` function samples an entirely new set of examples. It could for example be used to sample an entire new set of training data while training a classifier to prevent overfitting to the current examples. The `perc_normal` and `perc_distractors` arguments specify the percentage of non-transformed and distractor samples. Disctractor samples are transformed with default parameters and have a almost flat, noisy distribution over classes as their label. They can be used to slightly corrupt training data or to train a network to learn to be uncertain.

The amount of transformation can be set by changing the mean and standard deviation of each parameter:

```
set_scale(mean=1, std=1)
set_rotation(mean=0, std=math.pi / 2)
set_translation(mean=0, std=10)
```

This should be done *after* initialization of the class, but *before* calling `sample()` or `load_data()`.

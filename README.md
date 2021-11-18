# active_mnist
Small program that let's you sample from the MNIST handwritten digits dataset, but with random rotation, scaling and translation transformations applied.

The class can (for example) be used in the following way:
```
from active_mnist import Active_MNIST

sampler = Active_MNIST(image_path='./data/train-images.idx3-ubyte', label_path='./data/train-labels.idx1-ubyte')

original_img, transformed_img, labels, transforms = sampler.sample(10)
sampler.plot_example(original_img, transformed_img, labels, transforms)
```

The dataset is loaded when the class is initialized. You can then keep sampling new images from the dataset. 
The `sample()` function returns the original images, the transformed images, the labels and a vector containing the transformations. The transformations vector contains five numbers, which are: x-scaling, y-scaling, rotation, x-translation, y-translation. The minimum and maximum value of each transformation parameter can be specified as arguments to the `sample()` function. If one specific value is desired, set the minimum and maximum to the same value.
The `plot_example()` function takes the four arrays returned by the `sample()` function as input and plots one example by default. A larger number of examples can also be printed, as specified by the `number_examples` argument. 

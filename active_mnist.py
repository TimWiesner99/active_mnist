import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import idx2numpy
from PIL import Image

class Active_MNIST:

    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        
        self.images = idx2numpy.convert_from_file(self.image_path)
        self.labels = idx2numpy.convert_from_file(self.label_path)

    def sample(self, N, min_scale=0.5, max_scale=2, min_rotation= 0, max_rotation=math.pi, min_translation=0, max_translation=20):
        assert min_rotation >= 0
        assert max_rotation >= 0
        assert min_scale > 0

        def enlarge(image, shape=(56,56)):
            h, w = image.shape
            top  = (shape[0]-h)//2
            left = (shape[1]-w)//2

            out = np.zeros(shape)
            out[top:top+h, left:left+w] = image

            return out

        def transformation_matrix(transform=[1, 1, 0, 0, 0], shape=(56,56)):     
            #translate to center
            T_center = np.array([[1, 0, -shape[0]//2],
                                 [0, 1, -shape[1]//2],
                                 [0, 0, 1]])
            #scale
            T_scale = np.array([[transform[0], 0, 0],
                                [0, transform[1], 0],
                                [0, 0, 1]])
            #rotation
            T_rot = np.array([[np.cos(transform[2]), np.sin(transform[2]), 0],
                              [-np.sin(transform[2]), np.cos(transform[2]), 0],
                              [0, 0, 1]])
            #translation
            T_trans = np.array([[1, 0, transform[3]],
                                [0, 1, transform[4]],
                                [0, 0, 1]])
            #translate back to original space
            T_decenter = np.array([[1, 0, (shape[0]//2)],
                                   [0, 1, (shape[1]//2)],
                                   [0, 0, 1]])
            #combine matrices
            T = T_decenter @ T_trans @ T_rot @ T_scale @ T_center
            T_inv = np.linalg.inv(T)
            return T_inv

        index = np.random.randint(0, len(self.images), N)

        if not min_scale == max_scale:
            scales = np.random.randint(min_scale*100, max_scale*100, (N, 2))/100
        else:
            scales = np.zeros((N,2)) + min_scale

        if not min_rotation == max_rotation:
            rotations = np.random.randint(abs(min_rotation)*100, max_rotation*100, N) * np.random.choice([-1, 1], N) / 100 % (2*math.pi)
        else:
            rotations = np.zeros(N) + min_rotation

        if not min_translation == max_translation:
            translations = np.random.randint(min_translation, max_translation, (N, 2)) * np.random.choice([-1, 1], (N, 2))
        else:
            translations = np.zeros((N,2)) + min_translation

        trns = np.c_[scales, rotations, translations]
        lbl = np.array(self.labels[index])

        or_img = np.array([enlarge(image) for image in self.images[index]])
        trns_matrices = [transformation_matrix(t) for t in trns]

        trns_img = np.array([np.array(Image.fromarray(image).transform(image.shape, Image.AFFINE, data=trans_matrix.flatten()[:6], resample=Image.BILINEAR))
                   for image, trans_matrix in zip(or_img, trns_matrices)])

        return or_img, trns_img, lbl, trns

    def plot_example(self, original_images, transformed_images, labels, transforms, number_examples=1):
        assert len(original_images) == len(transformed_images) == len(labels) == len(transforms)
        assert len(original_images) >= number_examples

        for i in range(number_examples):
            fig, axs = plt.subplots(ncols=2, figsize=(6,3))
            axs[0].imshow(original_images[i], cmap=cm.gray_r)
            axs[0].set_title('Original MNIST Figure')
            axs[1].imshow(transformed_images[i], cmap=cm.gray_r)
            axs[1].set_title('Transformed MNIST Figure')
            description = f'Label = {labels[i]}\n' \
                           f'Scale = {transforms[i, 0:2]}\n' \
                           f'Rotation = {transforms[i, 2]:.2f} ({transforms[i, 2]/math.pi:.3f}π)\n' \
                           f'Translation = {transforms[i, 3:6]}'
            plt.text(0.95, 0.5, description, fontsize=14, transform=plt.gcf().transFigure)
            plt.show()
# MIT License
#
# Copyright (c) 2017 singleye
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import time
import matplotlib.pylab as plt
import numpy as np

matrix_edge_v1 = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, -4, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
]

matrix_edge_v2 = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
]

matrix_edge_v3 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, -4, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
]

matrix_edge_v4 = [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
]

matrix_edge_v5 = [
        [-0.125, -0.125, -0.125],
        [-0.125, 1, -0.125],
        [-0.125, -0.125, -0.125],
]

matrix_blur_v1 = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
]

matrix_blur_v2 = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
]

matrix_emboss_v1 = [
        [-1, -1, 0],
        [-1, 0, 1],
        [0, 1, 1]
]

matrix_sharpen_v1 = [
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1],
]

COLOR_SPACE_INVALID = 0
COLOR_SPACE_BW = 1
COLOR_SPACE_RGB = 2
COLOR_SPACE_RGBA = 3

CHANNEL_RED = 0
CHANNEL_GREEN = 1
CHANNEL_BLUE = 2
CHANNEL_ALPHA = 3

class Image(object):
    def __init__(self):
        self.img = None
        self.color_space = COLOR_SPACE_INVALID
        self.width = 0
        self.height = 0
        self.channels = 0

    def __identify_image(self, img):
        if len(img.shape) == 2:
            self.color_space = COLOR_SPACE_BW
            self.height, self.width = self.img.shape
            self.channels = 1
            print "Identify image: width[%d], height[%d], color: B&W" % \
                    (self.width, self.height)
            return

        self.height, self.width, self.channels = self.img.shape
        if self.channels == 3:
            self.color_space = COLOR_SPACE_RGB
            self.height, self.width, self.channels = self.img.shape
            print "Identify image: width[%d], height[%d], color: RGB" % \
                    (self.width, self.height)
        elif self.channels == 4:
            print "Identify image: width[%d], height[%d], color: RGBA" % \
                    (self.width, self.height)
            self.color_space = COLOR_SPACE_RGBA
        else:
            self.color_space = COLOR_SPACE_INVALID
            raise Exception("Unknown color space")

    def load(self, path):
        self.img = plt.imread(path)
        self.__identify_image(self.img)

    def load_data(self, data):
        self.img = data
        self.__identify_image(self.img)

    def show(self):
        print "Show image"
        plt.imshow(self.img)
        plt.axis('off')
        plt.show()

    def cut(self, x, y, width, height):
        if self.img is None:
            raise Exception("Image is not loaded!")

        h, w, c = self.img.shape
        if x > w or y > h:
            raise Exception("Coordinate exceeds the image size")

        if x + width > w:
            width = w - x
        if y + height > h:
            height = h - y

        self.img = self.img[y:(y+height), x:(x+width), :]

    def size(self):
        h, w, c = self.img.shape
        return w, h

    def convolution(self, kernel):
        """
        Create a new Image instance by applying the kernel
        """
        print "Run convolution transform"
        print "Start: %s" % time.ctime()

        k_height, k_width = kernel.shape
        n_width = self.width - k_width + 1
        n_height = self.height - k_height + 1

        if self.color_space == COLOR_SPACE_BW:
            new_img_data = np.zeros((n_height, n_width), dtype=self.img.dtype)
            channel_kernel = kernel
        elif self.color_space == COLOR_SPACE_RGB:
            new_img_data = np.zeros((n_height, n_width, 3), dtype=self.img.dtype)
            channel_kernel = np.zeros((k_height, k_width, 3))
            for c in range(3):
                channel_kernel[:,:,c] = kernel
        elif self.color_space == COLOR_SPACE_RGBA:
            # drop the alpha channel
            new_img_data = np.zeros((n_height, n_width, 3), dtype=self.img.dtype)
            channel_kernel = np.zeros((k_height, k_width, 3))
            for c in range(3):
                channel_kernel[:,:,c] = kernel
        else:
            print "Unknow color space"
            return None

        for y in range(n_height):
            for x in range(n_width):
                if self.color_space == COLOR_SPACE_RGBA:
                    new_img_data[y][x] = sum(sum(self.img[y:y+k_height, x:x+k_width,:3]*channel_kernel))
                else:
                    new_img_data[y][x] = sum(sum(self.img[y:y+k_height, x:x+k_width]*channel_kernel))

        imax = np.max(self.img)
        nmax = np.max(new_img_data)
        scale = 1.0*imax/nmax
        print "imax[{0}], nmax[{1}]".format(imax, nmax)
        print "Scale:", scale
        new_img_data = (new_img_data * scale).astype(self.img.dtype)

        print "End: %s" % time.ctime()

        new_image = Image()
        new_image.load_data(new_img_data)

        return new_image

    def get_channel(self, channel):
        print "Get new image from channel: %d" % channel
        if self.color_space == COLOR_SPACE_BW:
            raise Exception("Invalid image format")
        if channel > self.channels:
            raise Exception("Invalid channel")
        channel_data = np.zeros((self.height, self.width, 3), dtype=self.img.dtype)
        channel_data[:,:,channel] = self.img[:,:,channel]
        new_img = Image()
        new_img.load_data(channel_data)
        return new_img

    def detect_edge(self):
        print "Detecting edge"
        edge_kernel = np.array(matrix_edge_v5)
        new_img = self.convolution(edge_kernel)
        return new_img

    def blur(self):
        print "Blurring image"
        blur_kernel = np.array(matrix_blur_v2)
        new_img = self.convolution(blur_kernel)
        return new_img

    def emboss(self):
        print "Emboss image"
        emboss_kernel = np.array(matrix_emboss_v1)
        new_img = self.convolution(emboss_kernel)
        return new_img

    def sharpen(self):
        print "Sharpen image"
        sharpen_kernel = np.array(matrix_sharpen_v1)
        new_img = self.convolution(sharpen_kernel)
        return new_img

if __name__ == "__main__":
    img = Image()
    img.load(sys.argv[1])

    if img.color_space == COLOR_SPACE_BW:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        new_img = img.detect_edge()
        axs[0].imshow(img.img, cmap='gray')
        axs[1].imshow(new_img.img, cmap='gray')
    else:
        edge_img = img.detect_edge()
        blur_img = img.blur()

        red_img = img.get_channel(CHANNEL_RED)
        green_img = img.get_channel(CHANNEL_GREEN)
        blue_img = img.get_channel(CHANNEL_BLUE)

        fig, axs = plt.subplots(nrows=2, ncols=3)
        axs[0][0].imshow(img.img)
        axs[0][1].imshow(edge_img.img)
        axs[0][2].imshow(blur_img.img)
        axs[1][0].imshow(red_img.img)
        axs[1][1].imshow(green_img.img)
        axs[1][2].imshow(blue_img.img)
    plt.axis('off')
    plt.show()

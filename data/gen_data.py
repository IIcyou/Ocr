import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import random


class BackgroundImage(object):
    def __init__(self, width, height, model='RGB', color='black'):
        self.width = width
        self.height = height
        self.model = model
        self.color = color

    def do(self):
        data = Image.new(self.model, (self.width, self.height), self.color)
        # np_img = np.asarray(data, dtype='uint8')
        return data


class Font2Image(object):

    def __init__(self, font_path, font_size, text, bg_img):
        self.font_path = font_path
        self.font_size = font_size
        self.text = text
        self.bg_img = bg_img
        self.font = ImageFont.truetype(self.font_path, font_size, )
        text_size = self.font.getsize(self.text)

        self.shape = bg_img.size
        self.lt = self.get_lt(self.shape, text_size)
        print(self.lt, text_size, self.shape, text)

    def get_lt(self, shape, text_size):
        x = random.randint(0, shape[0] - text_size[0])
        y = random.randint(0, shape[1] - text_size[1])
        print(shape[0], text_size[0])
        return x, y

    def do(self):
        draw = ImageDraw.Draw(bg_img)
        # 白色字体
        draw.text(self.lt, self.text, (255, 255, 255), font=self.font)

        data = list(bg_img.getdata())

        np_img = np.asarray(data, dtype='uint8')
        np_img = np_img[:, 0]
        np_img = np_img.reshape((self.shape[1], self.shape[0]))

        return np_img


def add_noise(mask, low, high, channel=1, random=True, noise_factor=0.5):
    threshold_prob = noise_factor * 100
    c = np.count_nonzero(mask)
    # print('add_noise  c=', c, 'random=', random, 'channel=', channel)
    if random:
        nums = np.random.randint(low, high, c, dtype=np.uint8)
    else:
        nums = np.random.randint(low, high, channel, dtype=np.uint8)
        # print(nums)
        nums = nums[np.newaxis, :]
        nums = nums.repeat(int(c / channel), axis=0)
        nums = nums.reshape(nums.shape[0] * nums.shape[1])
    threshold_probs = np.random.randint(0, 100, c)
    vals = np.where(threshold_probs <= threshold_prob, nums, np.random.randint(low, high))
    # print(c, len(img[mask]), len(vals), np.count_nonzero(np.where(threshold_probs <= threshold_prob)))
    return mask, vals


def generalization_np(img, channel=1, random=True, bg_noise_factor=1, fg_noise_factor=1):
    shape = (img.shape[0], img.shape[1], channel)
    img = img.repeat(channel)
    img = img.reshape(shape)
    new_img = np.zeros(shape, dtype=np.uint8)

    #backgound
    mask, vals = add_noise(img < 50, 150, 255, random=random, noise_factor=bg_noise_factor, channel=channel)

    new_img[mask] = vals
    # print(new_img.shape, img.shape)
    #forgound
    mask, vals = add_noise(img > 50, 0, 100, random=random, noise_factor=fg_noise_factor, channel=channel)

    new_img[mask] = vals

    # np.place(new_img, img < 50, np.random.randint(150, 255))
    # np.place(new_img, img >50, np.random.randint(0, 100))
    return new_img


def generalization(img):
    shape = img.shape
    print(img.shape)
    new_img = np.zeros((shape[0], shape[1], 3))
    # new_img = np.zeros(shape)
    forgound = [np.random.randint(200, 255), np.random.randint(200, 255), np.random.randint(200, 255)]
    # backgound = [np.random.randint(0, 30), np.random.randint(0, 30), np.random.randint(0, 30)]
    backgound = [0, 0, 0]
    for y, row in enumerate(img):
        # print(row, y)
        for x, pixel in enumerate(row):
            if pixel != 0:
                print(pixel, pixel < 10)
            if pixel < 10:
                new_img[y, x] = backgound
            else:
                new_img[y, x] = forgound
    return new_img


if __name__ == '__main__':
    bg_image = BackgroundImage(640, 320)
    bg_img = bg_image.do()
    # cv2.imshow('bgimage', img)
    font2image = Font2Image('data/fonts/micross.ttf', 28, '998699', bg_img)
    img = font2image.do()
    cv2.imshow('img', img)
    # img1 = generalization(img)
    img1 = generalization_np(img, channel=3, random=True)
    # img1 = img1.astype(np.uint8)
    cv2.imshow('img1', img1)

    cv2.waitKey()

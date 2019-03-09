import numpy as np
import cv2


class PreProcessIm(object):
  def __init__(
      self,
      crop_prob=0,
      crop_ratio=1.0,
      resize_h_w=None,
      scale=True,
      im_mean=None,
      im_std=None,
      mirror_type=None,
      batch_dims='NCHW',
      prng=np.random,
      erase=True,
      erase_prob = 0.2, 
      sl = 0.02, 
      sh = 0.4, 
      r1 = 0.3, 
      mean=[0.0, 0.0, 0.0]
      ):
    """
    Args:
      crop_prob: the probability of each image to go through cropping
      crop_ratio: a float. If == 1.0, no cropping.
      resize_h_w: (height, width) after resizing. If `None`, no resizing.
      scale: whether to scale the pixel value by 1/255
      im_mean: (Optionally) subtracting image mean; `None` or a tuple or list or
        numpy array with shape [3]
      im_std: (Optionally) divided by image std; `None` or a tuple or list or
        numpy array with shape [3]. Dividing is applied only when subtracting
        mean is applied.
      mirror_type: How image should be mirrored; one of
        [None, 'random', 'always']
      batch_dims: either 'NCHW' or 'NHWC'. 'N': batch size, 'C': num channels,
        'H': im height, 'W': im width. PyTorch uses 'NCHW', while TensorFlow
        uses 'NHWC'.
      prng: can be set to a numpy.random.RandomState object, in order to have
        random seed independent from the global one
    """
    self.crop_prob = crop_prob
    self.crop_ratio = crop_ratio
    self.resize_h_w = resize_h_w
    self.scale = scale
    self.im_mean = im_mean
    self.im_std = im_std
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type
    self.check_batch_dims(batch_dims)
    self.batch_dims = batch_dims
    self.prng = prng

    r"""
    For image portion earasing
    """
    self.erase_prob = erase_prob
    self.mean = mean
    self.sl = sl
    self.sh = sh
    self.r1 = r1

  def __call__(self, im):
    return self.pre_process_im(im)

  @staticmethod
  def check_mirror_type(mirror_type):
    assert mirror_type in [None, 'random', 'always']

  @staticmethod
  def check_batch_dims(batch_dims):
    # 'N': batch size, 'C': num channels, 'H': im height, 'W': im width
    # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
    assert batch_dims in ['NCHW', 'NHWC']

  def set_mirror_type(self, mirror_type):
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type

  @staticmethod
  def rand_crop_im(im, new_size, prng=np.random):
    """Crop `im` to `new_size`: [new_w, new_h]."""
    if (new_size[0] == im.shape[1]) and (new_size[1] == im.shape[0]):
      return im
    h_start = prng.randint(0, im.shape[0] - new_size[1])
    w_start = prng.randint(0, im.shape[1] - new_size[0])
    im = np.copy(
      im[h_start: h_start + new_size[1], w_start: w_start + new_size[0], :])
    return im

  def pre_process_im(self, im):
    """Pre-process image.
    `im` is a numpy array with shape [H, W, 3], e.g. the result of
    matplotlib.pyplot.imread(some_im_path), or
    numpy.asarray(PIL.Image.open(some_im_path))."""

    # Randomly crop a sub-image.
    if ((self.crop_ratio < 1)
        and (self.crop_prob > 0)
        and (self.prng.uniform() < self.crop_prob)):
      print('C')
      h_ratio = self.prng.uniform(self.crop_ratio, 1)
      w_ratio = self.prng.uniform(self.crop_ratio, 1)
      crop_h = int(im.shape[0] * h_ratio)
      crop_w = int(im.shape[1] * w_ratio)
      im = self.rand_crop_im(im, (crop_w, crop_h), prng=self.prng)

    # Resize.
    if (self.resize_h_w is not None) \
        and (self.resize_h_w != (im.shape[0], im.shape[1])):
      im = cv2.resize(im, self.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)

    # scaled by 1/255.
    if self.scale:
      im = im / 255.

    # Subtract mean and scaled by std
    # im -= np.array(self.im_mean) # This causes an error:
    # Cannot cast ufunc subtract output from dtype('float64') to
    # dtype('uint8') with casting rule 'same_kind'
    if self.im_mean is not None:
      im = im - np.array(self.im_mean)
    if self.im_mean is not None and self.im_std is not None:
      im = im / np.array(self.im_std).astype(float)

    # May mirror image.
    mirrored = False
    if self.mirror_type == 'always' \
        or (self.mirror_type == 'random' and self.prng.uniform() > 0.5):
      im = im[:, ::-1, :]
      mirrored = True

    # Erase a random portion of image
    #if self.prng.uniform() < self.erase_prob:
    #  print('E')
    #  im = self.random_earase(im)


    # The original image has dims 'HWC', transform it to 'CHW'.
    if self.batch_dims == 'NCHW':
      im = im.transpose(2, 0, 1)

    return im, mirrored

  def random_earase(self, img):

    """ Randomly selects a rectangle region in an image and erases its pixels.
      'Random Erasing Data Augmentation' by Zhong et al.
      See https://arxiv.org/pdf/1708.04896.pdf
    Args:
       probability: The probability that the Random Erasing operation will be performed.
       sl: Minimum proportion of erased area against input image.
       sh: Maximum proportion of erased area against input image.
       r1: Minimum aspect ratio of erased area.
       mean: Erasing value. 
    """       
    import random
    import math

    # img [H W C]

    for _ in range(100):
        area = img.shape[1] * img.shape[0]
    
        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1/self.r1)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
            else:
                img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
            return img
    return img
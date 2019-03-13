from __future__ import print_function

import sys
import os

PACKAGE_PARENT = '../../tri_loss/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import torch
from torch.autograd import Variable
from torch.nn.parallel import DataParallel

import os.path as osp
from os.path import join as ospj

import numpy as np
import argparse

from dataset import create_dataset
from model.Model import Model

from utils.utils import time_str
from utils.utils import str2bool
from utils.utils import load_state_dict
from utils.utils import set_devices
from utils.utils import ReDirectSTD
from utils.utils import measure_time
from utils.distance import compute_dist
from utils.visualization import get_rank_list
from utils.visualization import save_rank_list_to_im

from apex import amp
from torch import optim

class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke'])

    parser.add_argument('--num_queries', type=int, default=16)
    parser.add_argument('--rank_list_size', type=int, default=10)

    parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
    parser.add_argument('--last_conv_stride', type=int, default=1,
                        choices=[1, 2])
    parser.add_argument('--normalize_feature', type=str2bool, default=False)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--ckpt_file', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')
    parser.add_argument('--opt-level', type=str, default='O0',
      choices=['O0', 'O1', 'O2', 'O3'])

    parser.add_argument('--net', type=str, default='mobilenetV2',
                      choices=['resnet50', 'shuffelnetV2', 'mobilenetV2'])



    args = parser.parse_args()

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    self.num_queries = args.num_queries
    self.rank_list_size = args.rank_list_size

    ###########
    # Dataset #
    ###########

    self.dataset = args.dataset
    self.prefetch_threads = 2

    # Image Processing

    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]

    self.test_mirror_type = None
    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    # The last block of ResNet has stride 2. We can set the stride to 1 so that
    # the spatial resolution before global pooling is doubled.
    self.last_conv_stride = args.last_conv_stride
    self.opt_level = args.opt_level
    self.net = args.net

    # Whether to normalize feature to unit length along the Channel dimension,
    # before computing distance
    self.normalize_feature = args.normalize_feature

    #######
    # Log #
    #######

    # If True, stdout and stderr will be redirected to file
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/visualize_rank_list',
        '{}'.format(self.dataset),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Model weights and optimizer states, for resuming.
    self.ckpt_file = args.ckpt_file
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    feat = self.model(ims)
    feat = feat.data.cpu().numpy().astype(np.float32)
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return feat


def main():
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  TVT, TMO = set_devices(cfg.sys_device_ids)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  ###########
  # Dataset #
  ###########

  test_set = create_dataset(**cfg.test_set_kwargs)

  #########
  # Model #
  #########

  model = Model(cfg.net, path_to_predefined='', pretrained=False, last_conv_stride=cfg.last_conv_stride)
  model.cuda()
  
  r'''
  This is compeletly useless, but since I used apex, and its optimization level
  has different effect on each layer of networ and optimizer is mandatory argument, I created this optimizer.
  '''
  optimizer = optim.Adam(model.parameters())
  
  model, optimizer = amp.initialize(model, optimizer,
                                    opt_level=cfg.opt_level,
                                    #loss_scale=cfg.loss_scale
                                )

  print(model)
  
  # Model wrapper
  model_w = DataParallel(model)

  # May Transfer Model to Specified Device.
  TMO([model])

  #####################
  # Load Model Weight #
  #####################

  # To first load weights to CPU
  map_location = (lambda storage, loc: storage)
  used_file = cfg.model_weight_file or cfg.ckpt_file
  loaded = torch.load(used_file, map_location=map_location)
  if cfg.model_weight_file == '':
    loaded = loaded['state_dicts'][0]
  load_state_dict(model, loaded)
  print('Loaded model weights from {}'.format(used_file))

  ###################
  # Extract Feature #
  ###################

  test_set.set_feat_func(ExtractFeature(model_w, TVT))

  with measure_time('Extracting feature...', verbose=True):
    feat, ids, cams, im_names, marks = test_set.extract_feat(True, verbose=True)

  #######################
  # Select Query Images #
  #######################

  # Fix some query images, so that the visualization for different models can
  # be compared.

  # Sort in the order of image names
  inds = np.argsort(im_names)
  feat, ids, cams, im_names, marks = \
    feat[inds], ids[inds], cams[inds], im_names[inds], marks[inds]

  # query, gallery index mask
  is_q = marks == 0
  is_g = marks == 1

  prng = np.random.RandomState(2)
  # selected query indices
  sel_q_inds = prng.permutation(range(np.sum(is_q)))[:cfg.num_queries]

  q_ids = ids[is_q][sel_q_inds]
  q_cams = cams[is_q][sel_q_inds]
  q_feat = feat[is_q][sel_q_inds]
  q_im_names = im_names[is_q][sel_q_inds]

  ####################
  # Compute Distance #
  ####################

  # query-gallery distance
  q_g_dist = compute_dist(q_feat, feat[is_g], type='euclidean')

  ###########################
  # Save Rank List as Image #
  ###########################


  q_im_paths = list()
  for n in q_im_names:
    if isinstance(n, bytes):
      n = n.decode("utf-8")
    q_im_paths.append(ospj(test_set.im_dir, n))
  
  save_paths = list()
  for n in q_im_names:
    if isinstance(n, bytes):
      n = n.decode("utf-8")
    save_paths.append(ospj(cfg.exp_dir, 'rank_lists', n))

  g_im_paths = list()
  for n in im_names[is_g]:
    if isinstance(n, bytes):
      n = n.decode("utf-8")
    g_im_paths.append(ospj(test_set.im_dir, n))

  for dist_vec, q_id, q_cam, q_im_path, save_path in zip(
      q_g_dist, q_ids, q_cams, q_im_paths, save_paths):

    rank_list, same_id = get_rank_list(
      dist_vec, q_id, q_cam, ids[is_g], cams[is_g], cfg.rank_list_size)

    save_rank_list_to_im(rank_list, same_id, q_im_path, g_im_paths, save_path)


if __name__ == '__main__':
  main()

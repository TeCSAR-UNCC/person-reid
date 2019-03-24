from __future__ import print_function

import sys
import os

PACKAGE_PARENT = '../../tri_loss/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import torch
import argparse
from model.Model import Model
from apex import amp
from torch import optim
#from torchsummary import summary



def save_models(net, check_point_file, model_file, image_h_w, onnx_file, opt_level):
    map_location = (lambda storage, loc: storage)
    data = torch.load(check_point_file, map_location=map_location)

    models = dict(data['state_dicts'][0])
    dummy_input = torch.randn(10, 3, image_h_w[0], image_h_w[1], device='cuda').half()
    model = Model(net, pretrained=False)
    model.load_state_dict(models)
    model.cuda()

    optimizer = optim.Adam(model.parameters())

    model, optimizer = amp.initialize(model, optimizer,
                                    opt_level=opt_level,
                                    #loss_scale=cfg.loss_scale
                                )




    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    torch.save(model.state_dict(), model_file)
    torch.onnx.export(model.base, dummy_input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

    #summary(model.base, input_size=(3, image_h_w[0], image_h_w[1]))

if __name__ == "__main__":
    pass_as_argument = True
    if pass_as_argument:
        parser = argparse.ArgumentParser()
        parser.add_argument('--check_point_file', type=str, default='')
        parser.add_argument('--model_file', type=str, default='')
        parser.add_argument('--onnx_file', type=str, default='')
        parser.add_argument('--net', type=str, default='shuffelnetV2',
                        choices=['resnet50', 'shuffelnetV2', 'mobilenetV2'])
        parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
        parser.add_argument('--opt-level', type=str, default='O0',
                        choices=['O0', 'O1', 'O2', 'O3'])

        args = parser.parse_args()
        check_point_file = args.check_point_file
        model_file = args.model_file
        onnx_file = args.onnx_file
        net = args.net
        image_h_w = args.resize_h_w
        opt_level = args.opt_level

    else:
        check_point_file = '/mnt/4tb/person-reid-triplet-loss-two-models-baseline/expt_res/ckpt.pth'
        model_file = '/mnt/4tb/person-reid-triplet-loss-two-models-baseline/expt_res/shflNet.pt'
        onnx_file = '/mnt/4tb/person-reid-triplet-loss-two-models-baseline/expt_res/shflNet.onnx'
        net = 'shuffelnetV2'
        opt_level = 'O0'
        image_h_w = (256, 128)
    
    save_models(check_point_file=check_point_file, model_file=model_file, net=net, image_h_w=image_h_w, onnx_file=onnx_file, opt_level=opt_level)


python3 ./tri_loss/utils/extract_weights.py --check_point_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/resnet50/ckpt.pth' \
--model_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/resnet50/resnet50.pt' \
--onnx_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/resnet50/resnet50.onnx' \
--net 'resnet50' \
--resize_h_w '(256, 128)' \
--opt-level 'O0'

#'/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/mobilenetV2_16bit/ckpt.pth'

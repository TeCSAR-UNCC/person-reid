python3 ./tri_loss/utils/extract_weights.py --check_point_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/mobilenet_16b/ckpt.pth' \
--model_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/mobilenet_16b/mobileNet.pt' \
--onnx_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/mobilenet_16b/mobileNet.onnx' \
--net 'mobilenetV2' \
--resize_h_w '(256, 128)'

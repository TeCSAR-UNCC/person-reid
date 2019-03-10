python3 ./tri_loss/utils/extract_weights.py --check_point_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/mobilenetV2/ckpt.pth' \
--model_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/mobilenetV2/mobileNet.pt' \
--onnx_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/mobilenetV2/mobileNet.onnx' \
--net 'mobilenetV2' \
--resize_h_w '(256, 128)'

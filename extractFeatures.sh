python2 ./script/experiment/infer_images_example.py \
-d '(0,1,)' \
--resize_h_w '(256, 128)' \
--ckpt_file '/mnt/4tb/person-reid-triplet-loss-two-models-baseline/expt_res/mobilenet/ckpt.pth' \
--saved_feature_mat_file './features' \
--image_dir '/home/mbaharan/shfs/TeCSAR/Datasets/ourEvaluationImage' \
--net 'mobilenetV2'

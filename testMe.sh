python3 script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset duke \
--last_conv_stride 1 \
--normalize_feature false \
--net mobilenetV2 \
--exp_dir ./res/mobilenet_16_not_training \
--model_weight_file /mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/mobilenetV2/mobileNet.pt \
--opt-level O3

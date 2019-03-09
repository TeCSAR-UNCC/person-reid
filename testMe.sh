python3 script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset combined \
--last_conv_stride 1 \
--normalize_feature false \
--net mobilenetV2 \
--exp_dir ./res/mobilenet \
--model_weight_file /mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/mobilenet/mobileNet.pt \
--opt-level O0

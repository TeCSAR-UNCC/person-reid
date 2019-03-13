python3 script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset combined \
--last_conv_stride 1 \
--normalize_feature false \
--net resnet50 \
--exp_dir ./res/mobilenet_16_not_training \
--model_weight_file /mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/expt_res/resnet50/resnet50.pt \
--opt-level O3

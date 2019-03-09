NET=shuffelnetV2
DIR=./expt_res/$NET

rm -rvf ./expt_res/$NET/tensorboard

python3 script/experiment/train.py \
-d '(1,)' \
--only_test false \
--dataset combined \
--normalize_feature false \
--trainset_part trainval \
--exp_dir $DIR \
--total_epochs 300 \
--steps_per_log 10 \
--epochs_per_val 5 \
--erase false \
--erase_prob 0.0 \
--crop_prob 0.0 \
--crop_ratio 1 \
--opt-level O0 \
--net  $NET \
--net_pretrained_path /mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/shufflenetv2_x1_69.402_88.374.pth.tar # It is not applicable for ResNet50


#--last_conv_stride 1 \ # Not useful in our case. 'market1501', 'cuhk03', 'duke', 'combined' #--lr_decay_type staircase \ #--resize_h_w '(224,224)' \
#--shuffle_net_pretrained_path /mnt/4tb/person-reid-triplet-loss-two-models-baseline/shufflenetv2_x1_69.402_88.374.pth.tar \ CUDA_VISIBLE_DEVICES=1 
#--net_pretrained_path /mnt/4tb/person-reid-triplet-loss-two-models-baseline-python3/mobilenet_v2.pth.tar \
#--base_lr 0.001 \shuffelnetV2

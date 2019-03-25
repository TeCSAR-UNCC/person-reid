NET=mobilenetV2
DIR=./expt_res/"$NET"_16bit

rm -rvf $DIR/tensorboard

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
--opt-level O2 \
--net  $NET \
--net_pretrained_path ./mobilenet_v2.pth.tar
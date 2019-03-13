#!/bin/bash
#last_conv_stride is only applicable for ResNet-50

NET=resnet50
NET_VER="$NET"_16bit
DIR=./expt_res/$NET_VER
Model=$NET.pt
Dataset=market1501

python3 script/experiment/visualize_rank_list.py \
-d '(0,)' \
--num_queries 16 \
--rank_list_size 10 \
--dataset $Dataset \
--normalize_feature false \
--exp_dir ./visu_dir/$NET_VER/$Dataset \
--model_weight_file $DIR/$Model \
--last_conv_stride 1 \
--opt-level "O2" \
--net $NET

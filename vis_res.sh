CUDA_VISIBLE_DEVICES=1 python2 script/experiment/visualize_rank_list.py \
-d '(0,)' \
--num_queries 16 \
--rank_list_size 10 \
--dataset  cuhk03 \
--normalize_feature false \
--exp_dir ./visu_dir/cuhk03 \
--ckpt_file ./expt_res/ckpt.pth

#--last_conv_stride is useless

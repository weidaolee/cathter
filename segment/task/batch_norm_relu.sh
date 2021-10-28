CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                    python train.py \
	                --prefix="batch_norm_relu" \
	                --gpu="0,1,2,3,4,5,6,7" \
	                --config_path="./task/batch_norm_relu.json" \
	                --train_path="./data/train_seg_tab.csv" \
	                --valid_path="./data/valid_seg_tab.csv" \
                    --checkpoint="./results/batch_norm_relu/checkpoints//model.pth" \

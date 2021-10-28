CUDA_VISIBLE_DEVICES=0,1,2,3 \
					python predict_segment.py \
                    --seg_prefix="pred_cvc"\
					--cls_prefix="train"\
					--task_name="stack_seg" \
	                --gpus="0,1,2,3" \
	                --config_path="./results/stack_seg/config.json" \
	                --data_path="./data/train_tab.csv" \
                    --checkpoint="./results/stack_seg/checkpoints/model.pth" \

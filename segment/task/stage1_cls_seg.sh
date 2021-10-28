CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
					python train.py \
                    --task_type="segmentation with catheter appearence" \
					--task_name="stage1_act" \
	                --gpus="0,1,2,3,4" \
	                --config_path="./task/stage1_cls_seg.json" \
	                --train_path="./data/train_seg_tab.csv" \
                    --valid_path="./data/valid_seg_tab.csv" \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
	   --prefix="test" \
	   --gpu="0,1,2,3,4,5,6,7" \
	   --config_path="config.json" \
	   --train_path="./data/train_seg_tab.csv" \
	   --valid_path="./data/valid_seg_tab.csv" \

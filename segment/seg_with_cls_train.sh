CUDA_VISIBLE_DEVICES="4,5,6,7"
python train.py \
	   --prefix="seg_with_cls" \
	   --gpu="4,5,6,7" \
	   --config_path="config.json" \
	   --train_path="./data/train_seg_tab.csv" \
	   --valid_path="./data/valid_seg_tab.csv" \
	   --checkpoint="./results/first_seg_with_cls/checkpoints/model_0020.pth" \

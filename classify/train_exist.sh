CUDA_VISIBLE_DEVICES=4,5,6,7
python train.py \
	   --prefix="exist" \
	   --gpu="4,5,6,7" \
	   --workers=24 \
	   --train_path="./data/train_tab.csv" \
	   --valid_path="./data/valid_tab.csv" \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task_type="classification with catheter seg input" \
--task_name="stage2" \
--gpus="0,1,2,3,4,5,6,7" \
--config_path="task/stage2.json" \
--train_path="./data/train_tab.csv" \
--valid_path="./data/valid_tab.csv" \
--checkpoint="./results/batch_norm_relu/checkpoints/new_model.pth" \

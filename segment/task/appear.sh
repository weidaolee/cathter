CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                    python train.py \
                    --task_type="segmentation with catheter appearence" \
                    --task_name="baseline_appear" \
                    --gpus="0,1,2,3,4,5,6,7" \
                    --config_path="task/appear.json" \
                    --train_path="./data/train_seg_tab.csv" \
                    --valid_path="./data/valid_seg_tab.csv" \
                    # --checkpoint="./results/stage1_act/checkpoints/model.pth" \

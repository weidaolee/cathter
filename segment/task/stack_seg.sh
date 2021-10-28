CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                    python train.py \
                    --task_type="segmentation with catheter appearence" \
                    --task_name="stack_seg" \
                    --gpus="0,1,2,3,4,5,6,7" \
                    --config_path="task/stack_seg.json" \
                    --train_path="./data/train_seg_tab.csv" \
                    --valid_path="./data/valid_seg_tab.csv" \
                    # --checkpoint="./results/appear/checkpoints/model.pth" \

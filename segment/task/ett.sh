CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                    python train.py \
                    --task_type="classification with catheter seg input" \
                    --task_name="ett_stack_softmax" \
                    --gpus="0,1,2,3,4,5,6,7" \
                    --config_path="task/ett.json" \
                    --train_path="./data/train_tab.csv" \
                    --valid_path="./data/valid_tab.csv" \
                    --checkpoint="./results/appear/checkpoints/model.pth" \

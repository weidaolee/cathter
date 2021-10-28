CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                    python train.py \
                    --task_type="classification with catheter seg input" \
                    --task_name="cvc_appear_sigmoid" \
                    --gpus="0,1,2,3,4,5,6,7" \
                    --config_path="task/cvc.json" \
                    --train_path="./data/train_tab.csv" \
                    --valid_path="./data/valid_tab.csv" \
                    --checkpoint="./results/appear/checkpoints/model.pth" \

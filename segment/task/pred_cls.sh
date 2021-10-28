CUDA_VISIBLE_DEVICES=0,1,2,3 \
					python predict_classify.py \
					--cls_prefix="valid" \
					--task_name="cvc_stack_sigmoid" \
	                --gpus="0,1,2,3" \
	                --config_path="./results/cvc_stack_sigmoid/config.json" \
	                --data_path="./data/valid_tab.csv" \
                    --checkpoint="./results/cvc_stack_sigmoid/checkpoints/model.pth" \

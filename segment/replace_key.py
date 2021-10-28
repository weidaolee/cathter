import torch
from collections import OrderedDict

ckpt_path = "./results/batch_norm_relu/checkpoints/model.pth"

state = torch.load(ckpt_path)

new_model_state = OrderedDict()

for k, v in state["model"].items():
    new_model_state[k.replace("exist", "appear")] = v

state["model"] = new_model_state

torch.save(state, "./results/batch_norm_relu/checkpoints/new_model.pth")

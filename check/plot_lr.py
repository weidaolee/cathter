import torch
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = nn.Linear(1, 1)
    lr = 1e-5 * 64 * 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = ReduceLROnPlateau(optimizer)
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=10,
    #     # T_mult=2,
    #     eta_min=1e-8,
    #     last_epoch=-1)

    # scheduler = GradualWarmupScheduler(optimizer,
    #                                    multiplier=1,
    #                                    total_epoch=10)

    # scheduler = CosineAnnealingLR(optimizer, T_max=10)
    step_size_down = 1470
    scheduler = CyclicLR(
        optimizer=optimizer,
        base_lr=1e-5,
        max_lr=lr,
        step_size_up=1,
        step_size_down=step_size_down,
        mode="exp_range",
        gamma=0.9998,
        cycle_momentum=False
    )

    # scheduler = OneCycleLR(
    #     optimizer=optimizer,
    #     max_lr=lr,
    #     epochs=100,
    #     steps_per_epoch=206,
    #     pct_start=0.1,
    #     anneal_strategy="cos",
    #     cycle_momentum=False,
    #     div_factor=1e-4,
    #     final_div_factor=25,
    # )

    max_itr = step_size_down * 10
    x = list(range(max_itr))
    y = []
    min_lr = 1024
    for i in range(1, max_itr + 1):
        optimizer.zero_grad()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']
        if lr < min_lr:
            min_lr = lr
        # print(i, lr)
        scheduler.step()
        y.append(scheduler.get_lr()[0])

    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel("lr")
    plt.title("learning rate's curve changes as epoch goes on!")
    plt.show()

import json

config = {
    "train": {
        "batch_size": 32,
        "prefetch": 1,
        "max_epoch": 50,
        "learning_rate": 1e-05,
        "warmup": 10,
        "weight_decay": 0.0005
    },
    "model": {
        "achitecture": "Unet",
        "classification_head": True,
        "parameters": {
            "encoder_name": "timm-res2net50_26w_8s",
            "encoder_depth": 5,
            "decoder_channels": [256, 128, 64, 32, 16],
            "in_channels": 9,
            "classes": 1,
            "decoder_attention_type": "scse"
        }
    }
}


s = json.dumps(config)

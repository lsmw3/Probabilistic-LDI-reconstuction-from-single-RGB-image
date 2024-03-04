import os
import yaml

exp_name = "exp1_autoencoder_kl_f4"

data = {
    "model": {
        "base_learning_rate": 4.5e-6,  # set to target_lr by starting main.py with '--scale_lr False'
        "target": "models.autoencoder.AutoencoderKL",
        "params": {
            "loss_path": "loss_logs/{now}_" + exp_name + "/",
            "embed_dim": 4,
            "monitor": "val/rec_loss",
            "image_key": "ldi",
            "lossconfig": {
                "target": "modules.losses.contperceptual.LPIPSWithDiscriminator",
                "params": {
                    "disc_start": 50001,
                    "kl_weight": 0.000001,
                    "disc_weight": 0.5
                },
            },
            "ddconfig": {
                "double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 64,
                "ch_mult": [1, 2, 4, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0
            }
        }
    },

    "data": {
        "target": "main.DataModuleFromConfig",
        "params": {
            "batch_size": 8,
            "num_workers": 8,
            "wrap": True,
            "train": {
                "target": "utils.dataloader.LayeredDepthImageTrain",
                "params": {
                    "apply_positional_encoding": False,
                    "apply_transform": True,
                    "flip": 0.5,
                    "multiply_data": 1,
                    "size": 256
                }
            },
            "validation": {
                "target": "utils.dataloader.LayeredDepthImageValidation",
                "params": {
                    "apply_positional_encoding": False,
                    "apply_transform": True,
                    "size": 256
                }
            }
        }
    },

    "lightning": {
        "callbacks": {
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 1000,
                    "max_images": 8,
                    "increase_log_steps": False,
                    "clamp": False
                }
            }
        },

        "trainer": {
            "benchmark": True,
            "accumulate_grad_batches": 2
        }
    }

}

yamlpath = "./configs/autoencoder-kl-f4.yaml"
with open(yamlpath, "w", encoding="utf-8") as f:
    yaml.dump(data, stream=f, allow_unicode=True)

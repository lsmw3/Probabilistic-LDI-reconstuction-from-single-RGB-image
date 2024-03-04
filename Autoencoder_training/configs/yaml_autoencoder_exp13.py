import os
import yaml

# copied from exp12
# added new loss coeff

exp_name = "exp13_autoencoder_ldi_aug_f4_noperceptual"

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
                    "kl_weight": 0.005,
                    "disc_weight": 0.5,
                    "perceptual_weight": 0.0,
                    "diff_loss_weight": 1,
                },
            },
            "ddconfig": {
                "double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 64,
                "ch_mult": [2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0
            }
        }
    },

    "data": {
        "target": "main.DataModuleFromConfig",
        "params": {
            "batch_size": 2,
            "num_workers": 8,
            "wrap": True,
            "train": {
                "target": "utils.dataloader.LayeredDepthImageTrain",
                "params": {
                    "apply_positional_encoding": False,
                    "apply_transform": True,
                    "flip": 0.5,
                    "multiply_data": 1,
                    "size": 256,
                    "ldi_mult_augmentation_prob": 0.5,
                    "ldi_mult_augmentation_factor": [0.75, 1.25]
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

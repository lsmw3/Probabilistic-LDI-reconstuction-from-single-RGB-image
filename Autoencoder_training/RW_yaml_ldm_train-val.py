import os
import yaml

exp_name = "train_on_larger_dataset"

data = {
    "model": {
        "base_learning_rate": 5.0e-5,  # set to target_lr by starting main.py with '--scale_lr False'
        "target": "models.diffusion.ddpm.LatentDiffusion",
        "params": {
            "linear_start": 0.0015,
            "linear_end": 0.0155,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "loss_type": "l2",
            "first_stage_key": "ldi",
            "cond_stage_key": "rgb",
            "image_size": 64,
            "channels": 4,
            "cond_stage_trainable": True,
            "concat_mode": True,
            "scale_by_std": True,
            "monitor": 'val/loss_simple_ema',
            "loss_path": f"loss_logs/{exp_name}",

            "scheduler_config": {
                "target": "lr_scheduler.LambdaLinearScheduler",
                "params": {
                    "warm_up_steps": [10000],
                    "cycle_lengths": [10000000000000],
                    "f_start": [1.e-6],
                    "f_max": [1.],
                    "f_min": [1.]
                }  # 10000 warmup steps
            },

            "unet_config": {
                "target": "modules.diffusionmodules.openaimodel.UNetModel",
                "params": {
                    "image_size": 64,
                    "in_channels": 8,
                    "out_channels": 4,
                    "model_channels": 160,
                    "attention_resolutions": [16, 8],  # 32, 16, 8, 4
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 2, 4],  # 32, 16, 8, 4, 2
                    "num_heads": 32,
                    "use_scale_shift_norm": True,
                    "resblock_updown": True
                }
            },

            "first_stage_config": {
                "target": "models.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 4,
                    "ckpt_path": "models/pre_trained_weights/kl-f4/train_on_7k.ckpt",
                    "ddconfig": {
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 64,
                        "ch_mult": [1, 2, 4], # latent code shape [4, 64, 64]
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0
                    },
                    "lossconfig": {
                        "target": "torch.nn.Identity"
                    }
                }
            },

            "cond_stage_config": {
                "target": "modules.encoders.modules.KL_Encoder",
                "params": {
                    "config": {
                        "double_z": False,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 64,
                        "attn_resolutions": [],
                        "ch_mult": [1, 2, 4],
                        "num_res_blocks": 2,
                        "dropout": 0.0
                    }
                }  # use the same encoder as first_stage_model, so the params should be the same, but no need for double z
            }
        }
    },

    "data": {
        "target": "main.DataModuleFromConfig",
        "params": {
            "batch_size": 4,
            "num_workers": 4,
            "wrap": False,
            "train": {
                "target": "utils.dataloader.LayeredDepthImageTrain",
                "params": {
                    "apply_positional_encoding": False,
                    "apply_ldi_augmentation": True,
                    "apply_rgb_transform": False,
                    "flip": 0.2,
                    "gray_scale": 0.2,
                    "gaussian_blur": 0.2,
                    "multiply_data": 1,
                    "size": 256,
                    "ldi_mult_augmentation_factor": [0.25, 2],
                    # "ignore_wrong_ldi_images": True
                }
            },
            "validation": {
                "target": "utils.dataloader.LayeredDepthImageValidation",
                "params": {
                    "apply_positional_encoding": False,
                    "size": 256,
                }
            }
        }
    },

    "lightning": {
        "callbacks": {
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 5000,
                    "max_images": 8,
                    "clamp": False,
                    "increase_log_steps": True,
                    "log_on_batch_idx": True
                }
            },
            "early_stop":{
                "target": "pytorch_lightning.callbacks.EarlyStopping",
                "params": {
                    "monitor": "val/loss_simple_ema",
                    "patience": 15,
                    "verbose": True,
                    "mode": "min",
                    "check_on_train_epoch_end": False
                }
            }
        },

        "trainer": {
            "benchmark": True,
            "check_val_every_n_epoch": 10
        }
    }
}

yamlpath = "./configs/ldi-kl-f4_train-val.yaml"
with open(yamlpath, "w", encoding="utf-8") as f:
    yaml.dump(data, stream=f, allow_unicode=True)

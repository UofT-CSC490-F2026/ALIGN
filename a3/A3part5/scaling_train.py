"""Select an initialization seed, then use the existing native-training wrapper."""
import os


def main():
    import torch
    import wandb
    import nanochat.common as common
    from A3part5.run_train import main as native_training

    seed = int(os.environ['A3_SCALING_SEED'])
    original_compute_init = common.compute_init
    original_wandb_init = wandb.init

    def compute_init(*args, **kwargs):
        result = original_compute_init(*args, **kwargs)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return result

    def wandb_init(*args, **kwargs):
        kwargs['group'] = 'scaling-study-v1'
        kwargs['config'] = dict(kwargs.get('config', {}), initialization_seed=seed)
        return original_wandb_init(*args, **kwargs)

    common.compute_init = compute_init
    wandb.init = wandb_init
    native_training()


if __name__ == '__main__':
    main()

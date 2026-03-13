import ast
import logging
import os
import os.path as op
import sys
from argparse import Namespace

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from omegaconf import DictConfig
import matplotlib.pyplot as plt

def _plot_and_save(array, figname, figsize=(12, 6), dpi=150):
    # --- 1. Sanitize Input ---
    # Ensure we are working with a CPU numpy array, not a GPU Tensor
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().float().numpy()
        
    shape = array.shape
    
    # --- 2. Debug Prints (Check your terminal!) ---
    print(f"Plotting {figname}")
    print(f"  Shape: {shape}")
    print(f"  Range: [{array.min():.4f}, {array.max():.4f}]")
    print(f"  Mean:  {array.mean():.4f}")
    
    if np.isnan(array).any():
        print("  WARNING: Array contains NaNs! Plot will be blank.")
        return # Stop here if data is corrupted

    # --- 3. Plotting Logic ---
    
    # CASE A: 1D (EOS Probability)
    if len(shape) == 1:
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(array)
        plt.xlabel("Frame")
        plt.ylabel("Probability")
        plt.ylim([0, 1])
        plt.tight_layout()
        os.makedirs(op.dirname(figname), exist_ok=True)
        plt.savefig(figname)
        plt.close()

    # CASE B: 2D Matrix (Tacotron style or generic heatmap)
    elif len(shape) == 2:
        plt.figure(figsize=figsize, dpi=dpi)
        # Transpose so Time (longer dim) is usually on X-axis
        if shape[0] > shape[1]: 
            array = array.T
            
        plt.imshow(array, aspect="auto", origin="lower", interpolation='nearest')
        plt.colorbar()
        plt.xlabel("Time / Frames")
        plt.ylabel("Input / Text")
        plt.tight_layout()
        os.makedirs(op.dirname(figname), exist_ok=True)
        plt.savefig(figname)
        plt.close()

    # CASE C: 4D Transformer Attention (Layers, Heads, Out, In)
    elif len(shape) == 4:
        # Create side-by-side comparison
        plt.figure(figsize=(14, 6), dpi=dpi)
        
        # 1. Global Average (Consensus)
        global_avg = np.mean(array, axis=(0, 1)) # Shape: (Out, In)
        
        # Transpose for (Text on Y, Time on X)
        global_avg = global_avg.T 
        
        plt.subplot(1, 2, 1)
        plt.imshow(global_avg, aspect="auto", origin="lower", interpolation='nearest')
        plt.title(f"Global Avg (Max: {global_avg.max():.2f})")
        plt.xlabel("Output (Time)")
        plt.ylabel("Input (Text)")
        plt.colorbar()

        # 2. Last Layer Average (Final Decision)
        last_layer = array[-1] # Shape: (Heads, Out, In)
        last_layer_avg = np.mean(last_layer, axis=0).T # Shape: (In, Out)
        
        plt.subplot(1, 2, 2)
        plt.imshow(last_layer_avg, aspect="auto", origin="lower", interpolation='nearest')
        plt.title(f"Last Layer Avg (Max: {last_layer_avg.max():.2f})")
        plt.xlabel("Output (Time)")
        # Share Y axis label with left plot implicitly
        plt.colorbar()

        plt.tight_layout()
        os.makedirs(op.dirname(figname), exist_ok=True)
        plt.savefig(figname)
        plt.close()

    else:
        print(f"Skipping plot for unsupported shape: {shape}")


# define function to calculate focus rate
# (see section 3.3 in https://arxiv.org/abs/1905.09263)
def _calculate_focus_rete(att_ws):
    if att_ws is None:
        # fastspeech case -> None
        return 1.0
    elif len(att_ws.shape) == 2:
        # tacotron 2 case -> (L, T)
        return float(att_ws.max(dim=-1)[0].mean())
    elif len(att_ws.shape) == 4:
        # transformer case -> (#layers, #heads, L, T)
        return float(att_ws.max(dim=-1)[0].mean(dim=-1).max())
    else:
        raise ValueError("att_ws should be 2 or 4 dimensional tensor.")


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)

    return _main(cfg, sys.stdout)


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("speecht5.generate_speech")

    utils.import_user_module(cfg.common)

    assert cfg.dataset.batch_size == 1, "only support batch size 1"
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if not use_cuda:
        logger.info("generate speech on cpu")

    # build task
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    # models, saved_cfg = checkpoint_utils.load_model_ensemble(
    #     utils.split_paths(cfg.common_eval.path),
    #     arg_overrides=overrides,
    #     task=task,
    #     suffix=cfg.checkpoint.checkpoint_suffix,
    #     strict=(cfg.checkpoint.checkpoint_shard_count == 1),
    #     num_shards=cfg.checkpoint.checkpoint_shard_count,
    # )
    # logger.info(saved_cfg)

    # --- START MODIFIED BLOCK ---

    # 1. Define the path to the model file
    model_path = utils.split_paths(cfg.common_eval.path)[0]
    logger.info("loading model(s) manually from {}".format(model_path))

    # 2. Manually load the checkpoint file using torch.load
    try:
        state = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint file at {model_path}: {e}")
        exit()

    # Since we cannot get the config from the checkpoint (it's missing metadata), 
    # we must rely on the configuration loaded by fairseq's config manager (cfg).
    saved_cfg = state['cfg']

    # 3. Initialize the model architecture using the loaded (or default) configuration
    # Note: task.build_model expects the model configuration part of saved_cfg
    model = task.build_model(state['cfg']['model'])

    # 4. Extract and clean model state dictionary
    # Use .get('model', state) to safely handle files that contain only the model dict
    model_state_dict = state.get('model', state)

    # Clean up state_dict keys for Hugging Face compatibility if necessary
    model_state_dict = {
        key.replace('_hf_text_encoder.', ''): value
        for key, value in model_state_dict.items()
    }

    # 5. Load the weights into the initialized model
    model.load_state_dict(model_state_dict, strict=True)
    models = [model] # Create the model ensemble (list) containing our single model

    logger.info("Successfully loaded model state dictionary.")
    # logger.info(saved_cfg) # We skip logging saved_cfg as it's the current cfg, not from the file

    # --- END MODIFIED BLOCK ---

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg['task'])

    # optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=None,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )
    
    for i, sample in enumerate(progress):
        if "net_input" not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        outs, _, attn = task.generate_speech(
            models, 
            sample["net_input"],
        )
        focus_rate = _calculate_focus_rete(attn)
        outs = outs.cpu().numpy()
        audio_name = op.basename(sample['name'][0])
        np.save(op.join(cfg.common_eval.results_path, audio_name.replace(".wav", "-feats.npy")), outs)

        logging.info(
            "{} (size: {}->{} ({}), focus rate: {:.3f})".format(
                sample['name'][0],
                sample['src_lengths'][0].item(),
                outs.shape[0],
                sample['dec_target_lengths'][0].item(), 
                focus_rate
            )
        )

        if i < 100 and attn is not None:
            import shutil
            demo_dir = op.join(op.dirname(cfg.common_eval.results_path), "demo")
            audio_dir = op.join(demo_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            shutil.copy(op.join(task.dataset(cfg.dataset.gen_subset).audio_root, sample['tgt_name'][0] if "tgt_name" in sample else sample['name'][0]), op.join(audio_dir, audio_name))
            att_dir = op.join(demo_dir, "att_ws")
            _plot_and_save(attn.cpu().numpy(), op.join(att_dir, f"{audio_name}_att_ws.png"))
            spec_dir = op.join(demo_dir, "spec")
            _plot_and_save(outs.T, op.join(spec_dir, f"{audio_name}_gen.png"))
            _plot_and_save(sample["target"][0].cpu().numpy().T, op.join(spec_dir, f"{audio_name}_ori.png"))


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()

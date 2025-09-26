""" Search for the optimal cfg weights for the given model.
    First using 10k samples to find the optimal value, then run on 50k samples to report.
    
    Options:
    - Use --clean-samples to delete NPZ files after evaluation (saves disk space)
    - Use --visualize-samples to create 10x10 grids of sample images from specific classes
"""
# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/sample/sample_c2i_ddp.py
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from omegaconf import OmegaConf
import json
from tqdm import tqdm
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import argparse
import sys
sys.path.append("./")
from RandAR.dataset.builder import build_dataset
from RandAR.utils.distributed import init_distributed_mode, is_main_process
from RandAR.dataset.augmentation import center_crop_arr
from RandAR.util import instantiate_from_config, load_safetensors
from RandAR.eval.fid import compute_fid


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def create_npz_from_samples_memory(samples_list, npz_path, total_samples):
    """
    Creates NPZ file directly from samples in memory.
    """
    if len(samples_list) > 0:
        # Stack all samples into a single numpy array
        samples = np.stack(samples_list)
        assert samples.shape == (total_samples, samples.shape[1], samples.shape[2], 3)
        np.savez(npz_path, arr_0=samples)
        print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def create_visualization_grid(tokenizer, gpt_model, args, device, class_indices, cfg_scale, samples_per_class=10):
    """
    Create a visualization grid with samples from specific classes.
    
    Args:
        tokenizer: The tokenizer model
        gpt_model: The GPT model
        args: Command line arguments
        device: GPU device
        class_indices: List of class indices to visualize
        cfg_scale: CFG scale to use for sampling
        samples_per_class: Number of samples per class (default 10)
    
    Returns:
        PIL Image of the grid
    """
    rank = dist.get_rank()
    
    if rank != 0:
        return None  # Only rank 0 creates visualization
    
    print(f"Creating visualization grid for classes: {class_indices}")
    
    # Save current random state to restore later
    current_rng_state = torch.get_rng_state()
    
    all_samples = []
    
    # Generate samples for each class with fixed seeds for reproducibility
    for idx, class_idx in enumerate(class_indices):
        # Set fixed seed for this class to ensure reproducible visualizations
        # Use a deterministic seed based on class index and global seed
        vis_seed = args.global_seed + (idx * 1000) + class_idx
        torch.manual_seed(vis_seed)
        
        print(f"Generating {samples_per_class} samples for class {class_idx} (seed: {vis_seed})")
        
        # Create batch of class indices
        c_indices = torch.full((samples_per_class,), class_idx, device=device)
        cfg_scales = (1.0, cfg_scale)
        
        # Generate indices
        indices = gpt_model.generate(
            cond=c_indices,
            token_order=None,
            cfg_scales=cfg_scales,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        # Decode to images
        samples = tokenizer.decode_codes_to_img(indices, args.image_size_eval)
        
        # Convert to numpy arrays and collect
        class_samples = []
        for sample in samples:
            if isinstance(sample, torch.Tensor):
                sample = sample.cpu().numpy()
            class_samples.append(sample)
        
        all_samples.extend(class_samples)
    
    # Create grid
    grid_size = len(class_indices)  # 10x10 for 10 classes
    image_size = args.image_size_eval
    
    # Create the grid image
    grid_image = Image.new('RGB', (grid_size * image_size, grid_size * image_size), color='white')
    
    # Place images in grid
    for i, sample in enumerate(all_samples):
        row = i // samples_per_class
        col = i % samples_per_class
        
        # Convert numpy array to PIL Image
        if sample.dtype != np.uint8:
            sample = (sample * 255).astype(np.uint8)
        
        sample_img = Image.fromarray(sample)
        
        # Paste into grid
        x = col * image_size
        y = row * image_size
        grid_image.paste(sample_img, (x, y))
    
    # Restore original random state to not affect main sampling
    torch.set_rng_state(current_rng_state)
    
    return grid_image


def save_visualization_grid(tokenizer, gpt_model, args, device, folder_name, cfg_scale):
    """
    Generate and save visualization grid if visualization classes are specified.
    
    Args:
        tokenizer: The tokenizer model
        gpt_model: The GPT model  
        args: Command line arguments
        device: GPU device
        folder_name: Base folder name for saving
        cfg_scale: CFG scale used for sampling
    """
    if not hasattr(args, 'visualize_samples') or not args.visualize_samples:
        return
    
    if dist.get_rank() != 0:
        return  # Only rank 0 handles visualization
    
    try:
        print("Generating visualization grid...")
        
        # Create visualization grid
        grid_image = create_visualization_grid(
            tokenizer=tokenizer,
            gpt_model=gpt_model,
            args=args,
            device=device,
            class_indices=args.visualize_samples,
            cfg_scale=cfg_scale,
            samples_per_class=10
        )
        
        if grid_image:
            # Create visualization filename
            vis_filename = f"{args.sample_dir}/{folder_name}.png"
            
            # Save the grid
            grid_image.save(vis_filename)
            print(f"Saved visualization grid to: {vis_filename}")
            
    except Exception as e:
        print(f"Warning: Failed to create visualization grid: {e}")


def sample_and_eval(tokenizer, gpt_model, cfg_scale, args, device, total_samples):
    # Use existing DDP setup from main():
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    torch.cuda.empty_cache()

    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert (
        total_samples % dist.get_world_size() == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert (
        samples_needed_this_gpu % args.per_proc_batch_size == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"

    folder_name = (
        f"{args.exp_name}-{args.ckpt_string_name}-size-{args.image_size}-size-{args.image_size_eval}-"
        f"cfg-{cfg_scale:.2f}-seed-{args.global_seed}-num-{total_samples}"
    )
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"Will create NPZ file at {sample_folder_dir}.npz")
    dist.barrier()

    iterations = int(samples_needed_this_gpu // args.per_proc_batch_size)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    global_batch_size = args.per_proc_batch_size * dist.get_world_size()
    
    # Store all generated indices and decoded samples in memory
    all_indices = []
    all_samples_memory = []
    
    cur_iter = 0
    for _ in pbar:
        c_indices = torch.randint(0, args.num_classes, (args.per_proc_batch_size,), device=device)
        cfg_scales = (1.0, cfg_scale)
    
        indices = gpt_model.generate(
            cond=c_indices,
            token_order=None,
            cfg_scales=cfg_scales,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        # Store indices for batch decoding later
        all_indices.append(indices)
        
        total += global_batch_size
        cur_iter += 1
        # I use this line to look at the initial images to check the correctness
        # comment this out if you want to generate more
        if args.debug:
            import pdb; pdb.set_trace()
    
    # Batch decode all generated indices using same batch size
    if rank == 0:
        print("Generation complete. Starting batch decoding...")
    
    # Decode and store samples in memory in batches of the same size as generation
    for batch_idx, indices_batch in enumerate(all_indices):
        # Decode this batch
        samples_batch = tokenizer.decode_codes_to_img(indices_batch, args.image_size_eval)
        
        # Store samples in CPU memory as numpy arrays
        for sample in samples_batch:
            # Ensure sample is on CPU as numpy array
            if isinstance(sample, torch.Tensor):
                sample = sample.cpu().numpy()
            all_samples_memory.append(sample)
    
    if rank == 0:
        print("Decoding complete. Samples stored in memory.")

    # Gather all samples from all processes
    dist.barrier()
    
    # Gather samples from all processes to rank 0 using CPU memory
    if rank == 0:
        print("Gathering samples from all processes to host memory...")
        all_samples_gathered = all_samples_memory.copy()
        
        for other_rank in range(1, dist.get_world_size()):
            # Receive samples from other ranks
            samples_from_rank = [None] * samples_needed_this_gpu
            for i in range(samples_needed_this_gpu):
                # Use CPU tensors for communication
                tensor_shape = torch.tensor([0, 0, 0], dtype=torch.int32, device='cpu')
                dist.recv(tensor_shape, src=other_rank)
                h, w, c = tensor_shape.tolist()
                
                # Create CPU tensor for receiving sample data
                sample_tensor = torch.zeros((h, w, c), dtype=torch.uint8, device='cpu')
                dist.recv(sample_tensor, src=other_rank)
                samples_from_rank[i] = sample_tensor.numpy()
            
            all_samples_gathered.extend(samples_from_rank)
        
        # Create NPZ directly from host memory
        npz_path = f"{sample_folder_dir}.npz"
        sample_path = create_npz_from_samples_memory(all_samples_gathered, npz_path, total_samples)
        print("Done.")
    else:
        # Send samples to rank 0 using CPU tensors
        for sample in all_samples_memory:
            # Send shape first using CPU tensor
            h, w, c = sample.shape
            shape_tensor = torch.tensor([h, w, c], dtype=torch.int32, device='cpu')
            dist.send(shape_tensor, dst=0)
            
            # Send sample data using CPU tensor
            sample_tensor = torch.from_numpy(sample).contiguous().to('cpu')
            dist.send(sample_tensor, dst=0)
        
        sample_path = None
    
    dist.barrier()

    # Create visualization grid if requested (before FID computation)
    save_visualization_grid(tokenizer, gpt_model, args, device, folder_name, cfg_scale)
    
    # Only rank 0 should compute FID since it has the complete sample data
    if dist.get_rank() == 0:
        fid, sfid, IS, precision, recall = compute_fid(args.ref_path, sample_path)
        
        # Clean up sample file if requested
        if sample_path and getattr(args, 'clean_samples', False):
            try:
                os.remove(sample_path)
                print(f"Cleaned up sample file: {sample_path}")
            except Exception as e:
                print(f"Warning: Failed to clean up sample file {sample_path}: {e}")
    else:
        # Other ranks return dummy values
        fid, sfid, IS, precision, recall = 0.0, 0.0, 0.0, 0.0, 0.0
    
    return fid, sfid, IS, precision, recall


def main(args):
    # Setup PyTorch:
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    config = OmegaConf.load(args.config)
    # create and load model
    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    tokenizer.load_state_dict(state_dict)

    # create and load gpt model
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    latent_size = args.image_size // args.downsample_size
    gpt_model = instantiate_from_config(config.ar_model).to(device=device, dtype=precision)
    model_weight = load_safetensors(args.gpt_ckpt)
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()

    # Create folder to save samples:
    ckpt_string_name = (
        os.path.basename(args.gpt_ckpt)
        .replace(".pth", "")
        .replace(".pt", "")
        .replace(".safetensors", "")
    )
    args.ckpt_string_name = ckpt_string_name

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()

    dist.barrier()
    
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples_search / global_batch_size) * global_batch_size)

    # CFG scales to be searched
    eval_results = {}
    
    result_file_name = (f"{args.results_path}/{args.exp_name}-{ckpt_string_name}-"
                        f"size-{args.image_size}-size-{args.image_size_eval}-search.json")

    # Create results directory if it doesn't exist
    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(result_file_name), exist_ok=True)

    # Skip search phase if cfg_optimal_scale is provided
    if args.cfg_optimal_scale is not None:
        optimal_cfg_scale = args.cfg_optimal_scale
        print(f"Using provided optimal CFG scale: {optimal_cfg_scale:.2f}")
    else:
        cfg_scales_search = args.cfg_scales_search.split(",")
        cfg_scales_search = [float(cfg_scale) for cfg_scale in cfg_scales_search]
        cfg_scales_interval = float(args.cfg_scales_interval)
        cfg_scales_list = np.arange(cfg_scales_search[0], cfg_scales_search[1] + 1e-4, cfg_scales_interval)
        print(f"CFG scales to be searched: {cfg_scales_list}")

        # run throught the CFG scales
        for cfg_scale in cfg_scales_list:
            fid, sfid, IS, precision, recall = sample_and_eval(
                tokenizer, gpt_model, cfg_scale, args, device, total_samples)
            
            # Only rank 0 processes results and saves to file
            if dist.get_rank() == 0:
                eval_results[f"{cfg_scale:.2f}"] = {
                    "fid": fid,
                    "sfid": sfid,
                    "IS": IS,
                    "precision": precision,
                    "recall": recall
                }
                print(f"Eval results for CFG scale {cfg_scale:.2f}: {eval_results[f'{cfg_scale:.2f}']}")

                with open(result_file_name, "w") as f:
                    json.dump(eval_results, f)
        
        # Only rank 0 determines optimal CFG scale
        if dist.get_rank() == 0:
            optimal_cfg_scale = float(min(eval_results, key=lambda x: eval_results[x]["fid"]))
        else:
            optimal_cfg_scale = 0.0  # Dummy value for non-rank-0 processes
        
        # Broadcast optimal CFG scale to all ranks
        optimal_cfg_tensor = torch.tensor([optimal_cfg_scale], dtype=torch.float32, device=device)
        dist.broadcast(optimal_cfg_tensor, src=0)
        optimal_cfg_scale = optimal_cfg_tensor.item()

    # report the results
    total_samples = int(math.ceil(args.num_fid_samples_report / global_batch_size) * global_batch_size)
    fid, sfid, IS, precision, recall = sample_and_eval(
        tokenizer, gpt_model, optimal_cfg_scale, args, device, total_samples)
    
    # Only rank 0 handles final reporting and file saving
    if dist.get_rank() == 0:
        print(f"Optimal CFG scale: {optimal_cfg_scale:.2f}")
        print(f"Eval results for optimal CFG scale: {fid, sfid, IS, precision, recall}")
        eval_results[f"{optimal_cfg_scale:.2f}-report"] = {
            "fid": fid,
            "sfid": sfid,
            "IS": IS,
            "precision": precision,
            "recall": recall
        }

        with open(result_file_name, "w") as f:
            json.dump(eval_results, f)
    
    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sample results
    parser.add_argument("--config", type=str, default="configs/randar/randar_xl_0.7b.yaml")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i"], default="c2i")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input",)
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[128, 256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scales-search", type=str, default="2.0, 8.0")
    parser.add_argument("--cfg-scales-interval", type=float, default=0.2)
    parser.add_argument("--cfg-optimal-scale", type=float, default=None, help="If specified, skip search phase and use this CFG scale directly")
    parser.add_argument("--sample-dir", type=str, default="/tmp")
    parser.add_argument("--num-inference-steps", type=int, default=88)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples-search", type=int, default=10000)
    parser.add_argument("--num-fid-samples-report", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--ref-path", type=str, default="/tmp/VIRTUAL_imagenet256_labeled.npz")
    # output results
    parser.add_argument("--results-path", type=str, default="./results")
    # cleanup options
    parser.add_argument("--clean-samples", action="store_true", default=False, help="Delete sample NPZ files after evaluation")
    # visualization options
    parser.add_argument("--visualize-samples", type=int, nargs="*", 
                       default=[555, 812, 207, 417, 487, 416, 981, 537, 0, 801],
                       help="Class indices to visualize in a 10x10 grid (10 samples per class). Default: fire engine, space shuttle, golden retriever, hot air balloon, cell phone, balance beam, baseball player, dog sled, tench, snorkel")
    args = parser.parse_args()
    main(args)

""" Search for the optimal cfg weights for the given model.
    First using 10k samples to find the optimal value, then run on 50k samples to report.
    
    GCS Support:
    - Set --sample-dir to a GCS path (e.g., gs://my-bucket/samples/) to upload NPZ files to GCS
    - Use --gcs-project-id to specify a specific GCS project (optional)
      - When set with GCS sample-dir, also uploads results JSON to the same bucket at {bucket}/{results-path}/
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
from PIL import Image
import numpy as np
import math
import argparse
import sys
import threading
import queue
import time
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not available. GCS functionality will be disabled.")
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


def init_gcs_client(gcs_project_id=None):
    """
    Initialize Google Cloud Storage client.
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not available. Please install it with: pip install google-cloud-storage")
    
    if gcs_project_id:
        client = storage.Client(project=gcs_project_id)
    else:
        # Use default credentials and project
        client = storage.Client()
    return client


def upload_to_gcs(local_path, gcs_bucket_name, gcs_blob_name, gcs_project_id=None):
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        local_path: Local file path to upload
        gcs_bucket_name: GCS bucket name
        gcs_blob_name: GCS blob (object) name
        gcs_project_id: Optional GCS project ID
    
    Returns:
        GCS URI of the uploaded file
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not available. Please install it with: pip install google-cloud-storage")
    
    client = init_gcs_client(gcs_project_id)
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_blob_name)
    
    print(f"Uploading {local_path} to gs://{gcs_bucket_name}/{gcs_blob_name}")
    blob.upload_from_filename(local_path)
    gcs_uri = f"gs://{gcs_bucket_name}/{gcs_blob_name}"
    print(f"Successfully uploaded to {gcs_uri}")
    
    return gcs_uri


def is_gcs_path(path):
    """
    Check if a path is a GCS path (starts with gs://).
    """
    return path.startswith("gs://")


def parse_gcs_path(gcs_path):
    """
    Parse a GCS path to extract bucket name and blob name.
    
    Args:
        gcs_path: GCS path like "gs://bucket-name/path/to/file"
    
    Returns:
        tuple: (bucket_name, blob_name)
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with 'gs://'")
    
    path_without_prefix = gcs_path[5:]  # Remove "gs://"
    parts = path_without_prefix.split("/", 1)
    
    if len(parts) < 2:
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must include bucket and object name")
    
    bucket_name = parts[0]
    blob_name = parts[1]
    
    return bucket_name, blob_name


def upload_results_to_gcs(local_results_path, gcs_bucket_name, results_path, gcs_project_id=None):
    """
    Upload results JSON file to GCS bucket with the same directory structure.
    
    Args:
        local_results_path: Local path to the results JSON file
        gcs_bucket_name: GCS bucket name (extracted from sample_dir)
        results_path: Original results_path argument (e.g., "./results")
        gcs_project_id: Optional GCS project ID
    
    Returns:
        GCS URI of the uploaded results file
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not available. Please install it with: pip install google-cloud-storage")
    
    # Get the relative path from the results directory
    results_filename = os.path.basename(local_results_path)
    
    # Clean up the results_path (remove leading ./ if present)
    clean_results_path = results_path.lstrip("./")
    
    # Create the GCS blob name: results_path/filename
    gcs_blob_name = f"{clean_results_path}/{results_filename}"
    
    # Upload the file
    client = init_gcs_client(gcs_project_id)
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_blob_name)
    
    print(f"Uploading results to gs://{gcs_bucket_name}/{gcs_blob_name}")
    blob.upload_from_filename(local_results_path)
    gcs_uri = f"gs://{gcs_bucket_name}/{gcs_blob_name}"
    print(f"Successfully uploaded results to {gcs_uri}")
    
    return gcs_uri


class GCSUploadManager:
    """
    Manager for handling background GCS uploads using threading.
    """
    def __init__(self, gcs_project_id=None):
        self.gcs_project_id = gcs_project_id
        self.upload_queue = queue.Queue()
        self.upload_threads = []
        self.upload_results = {}  # Track upload results
        self.active_uploads = 0
        self.lock = threading.Lock()
        self.max_workers = 4  # Number of concurrent upload threads
        
    def _upload_worker(self):
        """Worker function for upload threads."""
        while True:
            try:
                task = self.upload_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                    
                upload_type, file_path, gcs_bucket_name, blob_name = task
                
                with self.lock:
                    self.active_uploads += 1
                
                try:
                    if upload_type == "npz":
                        gcs_uri = upload_to_gcs(
                            local_path=file_path,
                            gcs_bucket_name=gcs_bucket_name,
                            gcs_blob_name=blob_name,
                            gcs_project_id=self.gcs_project_id
                        )
                    elif upload_type == "results":
                        gcs_uri = upload_to_gcs(
                            local_path=file_path,
                            gcs_bucket_name=gcs_bucket_name,
                            gcs_blob_name=blob_name,
                            gcs_project_id=self.gcs_project_id
                        )
                    
                    with self.lock:
                        self.upload_results[file_path] = {"success": True, "gcs_uri": gcs_uri}
                        
                except Exception as e:
                    with self.lock:
                        self.upload_results[file_path] = {"success": False, "error": str(e)}
                        print(f"Background upload failed for {file_path}: {e}")
                
                finally:
                    with self.lock:
                        self.active_uploads -= 1
                    self.upload_queue.task_done()
                    
            except queue.Empty:
                continue
                
    def start_workers(self):
        """Start the upload worker threads."""
        for _ in range(self.max_workers):
            thread = threading.Thread(target=self._upload_worker, daemon=True)
            thread.start()
            self.upload_threads.append(thread)
    
    def queue_npz_upload(self, local_path, gcs_bucket_name, blob_name):
        """Queue an NPZ file for background upload."""
        self.upload_queue.put(("npz", local_path, gcs_bucket_name, blob_name))
        print(f"Queued NPZ upload: {local_path} -> gs://{gcs_bucket_name}/{blob_name}")
    
    def queue_results_upload(self, local_path, gcs_bucket_name, blob_name):
        """Queue a results file for background upload."""
        self.upload_queue.put(("results", local_path, gcs_bucket_name, blob_name))
        print(f"Queued results upload: {local_path} -> gs://{gcs_bucket_name}/{blob_name}")
    
    def wait_for_all_uploads(self, timeout=3600):
        """Wait for all uploads to complete."""
        print("Waiting for all GCS uploads to complete...")
        start_time = time.time()
        
        while True:
            with self.lock:
                queue_size = self.upload_queue.qsize()
                active = self.active_uploads
            
            if queue_size == 0 and active == 0:
                print("All GCS uploads completed!")
                break
                
            if time.time() - start_time > timeout:
                print(f"Warning: Upload timeout after {timeout}s. Some uploads may not have completed.")
                break
                
            print(f"Uploads in progress: {active} active, {queue_size} queued")
            time.sleep(5)
    
    def shutdown(self):
        """Shutdown the upload manager."""
        # Signal workers to stop
        for _ in range(len(self.upload_threads)):
            self.upload_queue.put(None)
        
        # Wait for threads to finish
        for thread in self.upload_threads:
            thread.join(timeout=10)
    
    def get_upload_summary(self):
        """Get a summary of upload results."""
        with self.lock:
            total = len(self.upload_results)
            successful = sum(1 for result in self.upload_results.values() if result["success"])
            failed = total - successful
            return {"total": total, "successful": successful, "failed": failed}


def save_results_with_gcs_upload(eval_results, result_file_name, args, gcs_manager=None):
    """
    Save results JSON file locally and optionally upload to GCS.
    
    Args:
        eval_results: Dictionary of evaluation results
        result_file_name: Local path for the results file
        args: Command line arguments
        gcs_manager: Optional GCSUploadManager for background uploads
    """
    # Save results locally
    with open(result_file_name, "w") as f:
        json.dump(eval_results, f)
    
    # Upload to GCS if project ID is specified
    if args.gcs_project_id:
        # Check if we have a GCS sample directory to extract bucket from
        original_sample_dir = getattr(args, '_original_sample_dir', args.sample_dir)
        if is_gcs_path(original_sample_dir):
            try:
                # Extract bucket name from the original sample directory
                gcs_bucket_name, gcs_base_path = parse_gcs_path(original_sample_dir)
                
                # Create blob name for results
                results_filename = os.path.basename(result_file_name)
                clean_results_path = args.results_path.lstrip("./")
                blob_name = f"{clean_results_path}/{results_filename}"
                
                # Queue for background upload
                gcs_manager.queue_results_upload(result_file_name, gcs_bucket_name, blob_name)
            except Exception as e:
                print(f"Warning: Failed to queue results upload to GCS: {e}")
                print("Results saved locally only.")
        else:
            print("Note: gcs-project-id specified but sample-dir is not a GCS path.")
            print("To upload results to GCS, specify a GCS path for --sample-dir (e.g., gs://my-bucket/samples/)")


def sample_and_eval(tokenizer, gpt_model, cfg_scale, args, device, total_samples, gcs_manager=None):
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

    # Handle GCS upload if original sample_dir is a GCS path
    original_sample_dir = getattr(args, '_original_sample_dir', args.sample_dir)
    if dist.get_rank() == 0 and sample_path and is_gcs_path(original_sample_dir):
        try:
            # Parse the original GCS path to get bucket and determine the blob name
            gcs_bucket_name, gcs_base_path = parse_gcs_path(original_sample_dir)
            
            # Create the blob name from the folder structure
            blob_name = f"{gcs_base_path.rstrip('/')}/{folder_name}.npz"
            
            
            # Queue for background upload
            gcs_manager.queue_npz_upload(sample_path, gcs_bucket_name, blob_name)
            print(f"NPZ file will be uploaded to: gs://{gcs_bucket_name}/{blob_name}")
            
        except Exception as e:
            print(f"Warning: Failed to queue/upload to GCS: {e}")
            print("Continuing with local file...")

    # Only rank 0 should compute FID since it has the complete sample data
    if dist.get_rank() == 0:
        fid, sfid, IS, precision, recall = compute_fid(args.ref_path, sample_path)
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

    # Validate GCS configuration and create local sample directory
    if rank == 0:
        if is_gcs_path(args.sample_dir):
            if not GCS_AVAILABLE:
                raise ImportError("GCS path specified but google-cloud-storage is not available. Please install it with: pip install google-cloud-storage")
            
            # Validate GCS path format
            try:
                bucket_name, base_path = parse_gcs_path(args.sample_dir)
                print(f"GCS configuration: bucket={bucket_name}, base_path={base_path}")
                
                # Test GCS connectivity
                client = init_gcs_client(args.gcs_project_id)
                bucket = client.bucket(bucket_name)
                if not bucket.exists():
                    print(f"Warning: GCS bucket '{bucket_name}' may not exist or is not accessible")
                else:
                    print(f"Successfully connected to GCS bucket: {bucket_name}")
                    
            except Exception as e:
                raise ValueError(f"Invalid GCS configuration: {e}")
            
            # Create a local temporary directory for NPZ files before uploading
            local_sample_dir = "/tmp/randar_samples"
            os.makedirs(local_sample_dir, exist_ok=True)
            print(f"Using local temporary directory: {local_sample_dir}")
            # Store original GCS path for later use
            args._original_sample_dir = args.sample_dir
            args.sample_dir = local_sample_dir
        else:
            # Regular local directory
            os.makedirs(args.sample_dir, exist_ok=True)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()

    dist.barrier()
    
    # Initialize GCS upload manager if needed
    gcs_manager = None
    if dist.get_rank() == 0 and args.gcs_project_id:
        original_sample_dir = getattr(args, '_original_sample_dir', args.sample_dir)
        if is_gcs_path(original_sample_dir):
            gcs_manager = GCSUploadManager(gcs_project_id=args.gcs_project_id)
            gcs_manager.start_workers()
            print("Started GCS upload manager for background uploads")
    
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
                tokenizer, gpt_model, cfg_scale, args, device, total_samples, gcs_manager)
            
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

                save_results_with_gcs_upload(eval_results, result_file_name, args, gcs_manager)
        
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
        tokenizer, gpt_model, optimal_cfg_scale, args, device, total_samples, gcs_manager)
    
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

        save_results_with_gcs_upload(eval_results, result_file_name, args, gcs_manager)
        
        # Wait for all GCS uploads to complete before finishing
        if gcs_manager:
            gcs_manager.wait_for_all_uploads()
            summary = gcs_manager.get_upload_summary()
            print(f"GCS Upload Summary: {summary['successful']}/{summary['total']} successful uploads")
            gcs_manager.shutdown()
    
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
    parser.add_argument("--sample-dir", type=str, default="temp")
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
    # GCS configuration
    parser.add_argument("--gcs-project-id", type=str, default="flowmo", help="GCS project ID (optional, uses default if not specified)")
    args = parser.parse_args()
    main(args)

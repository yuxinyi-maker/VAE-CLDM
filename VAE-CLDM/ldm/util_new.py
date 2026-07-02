import importlib
import torch
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont


def log_txt_as_img(wh, xc, size=10):
    """
    Render text log content into image format for visualization

    Args:
        wh: tuple (width, height),specifies width and height of generated image
        xc: list of strings,each string represents text content of one batch
        size: font size, default is 10

    Returns:
        torch.Tensor: rendered text image tensor,shape is [batch, channel, height, width]
    """
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    """
    Check whether input is a feature map(4D tensor with more than 3 channels)

    Args:
        x: input tensor

    Returns:
        bool: return True if feature map, otherwise return False
    """
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    """
    Check whether input is an image tensor(4D tensor with 3 or 1 channels)

    Args:
        x: input tensor

    Returns:
        bool: return True if image, otherwise return False
    """
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def is3drock(x):
    """
    Check whether input is a 3D rock data tensor(5D tensor with 1 channel)

    Args:
        x: input tensor

    Returns:
        bool: return True if 3D rock data, otherwise return False
    """
    if not isinstance(x, torch.Tensor):
        return False
    # shape should be (batch, channel, depth, height, width),channel count of 1 indicates binary rock data
    return (len(x.shape) == 5) and (x.shape[1] == 1)


def exists(x):
    """
    Check whether variable exists(not None)

    Args:
        x: input variable

    Returns:
        bool: return True if exists, otherwise return False
    """
    return x is not None


def default(val, d):
    """
    Provide default value; return val if it exists, otherwise return d

    Args:
        val: value to check
        d: default value or function returning default value

    Returns:
        value of val or d
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    Average over all non-batch dimensions, suitable for tensors of any dimension

    Args:
        tensor: input tensor

    Returns:
        torch.Tensor: tensor averaged over all non-batch dimensions
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    """
    Compute number of model parameters

    Args:
        model: PyTorch model
        verbose: whether to print parameter count information

    Returns:
        int: total number of model parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    """
    Instantiate object from config dictionary

    Args:
        config: config dictionary containing 'target' key

    Returns:
        instantiated object

    Raises:
        KeyError: if config is missing 'target' key
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """
    Get class or function object from string path

    Args:
        string: full path string of class or function("module.submodule.ClassName")
        reload: whether to reload module

    Returns:
        class or function object
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    """
    Helper function for parallel data prefetching, executed in worker process

    Args:
        func: function to execute
        Q: queue used to return results
        data: data to process
        idx: worker process index
        idx_to_fn: whether to pass index to function
    """
    # Run prefetch task
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    """
    Parallel data prefetch function for accelerating data loading and processing

    Args:
        func: function to execute in parallel
        data: data to process,can be ndarray or iterable object
        n_proc: number of processes/threads to use
        target_data_type: target data type,"ndarray""list"
        cpu_intensive: whether task is CPU-intensive, determining whether to use processes or threads
        use_worker_id: whether to pass worker ID to function

    Returns:
        processed data, type specified by target_data_type

    Raises:
        ValueError: data type mismatch
        TypeError: unsupported data type
    """
    # target data type
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    # Choose whether to use processes or threads
    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread

    # Prepare arguments
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]

    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # Start parallel processing
    print(f"Start prefetching...")
    import time
    start = time.time()
    gather_res = [[] for _ in range(n_proc)]

    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # Get results
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()
        raise e

    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    # Organize and return results
    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res


def load_rock_data(file_path, normalize=True):
    """
    Load 3D rock data file(.npyformat)

    Args:
        file_path: .npyfile path
        normalize: whether to normalize data to [0,1] range

    Returns:
        torch.Tensor: loaded rock data tensor
    """
    data = np.load(file_path)
    data = torch.from_numpy(data).float()

    if normalize:
        # Ensure data is in [0,1] range
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)

    # Add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
    if len(data.shape) == 3:
        data = data.unsqueeze(0).unsqueeze(0)
    elif len(data.shape) == 4:
        data = data.unsqueeze(0)

    return data


def create_rock_conditioning(rock_type, porosity):
    """
    Create rock condition vector combining rock type and porosity information

    Args:
        rock_type: rock type string("limestone", "sandstone")
        porosity: porosity value(0-1floating-point number between)

    Returns:
        dict: dictionary containing text condition and porosity condition
    """
    # Create text condition
    text_condition = f"{rock_type} rock with porosity {porosity:.3f}"

    # Create porosity condition tensor
    porosity_tensor = torch.tensor([porosity], dtype=torch.float32)

    return {
        'text': text_condition,
        'porosity': porosity_tensor
    }


def preprocess_rock_batch(rock_data, conditions, device='cuda'):
    """
    Preprocess rock data batch for model training

    Args:
        rock_data: rock data tensor
        conditions: condition information list
        device: target device

    Returns:
        tuple: (processed data, condition tensor)
    """
    # Ensure data is on the correct device
    rock_data = rock_data.to(device)

    # Process condition information
    text_conditions = []
    porosity_conditions = []

    for cond in conditions:
        text_conditions.append(cond['text'])
        porosity_conditions.append(cond['porosity'])

    # Convert porosity condition to tensor
    porosity_tensor = torch.stack(porosity_conditions).to(device)

    return rock_data, {
        'text': text_conditions,
        'porosity': porosity_tensor
    }


def save_rock_data(data, file_path, threshold=0.5):
    """
    Save generated rock data as .npy file

    Args:
        data: rock data tensor
        file_path: save path
        threshold: binarization threshold
    """
    # Convert to numpy array and remove batch and channel dimensions
    if len(data.shape) == 5:
        data = data.squeeze(0).squeeze(0)  # (1, 1, D, H, W) -> (D, H, W)
    elif len(data.shape) == 4:
        data = data.squeeze(0)  # (1, D, H, W) -> (D, H, W)

    # Binarization processing
    data_np = data.cpu().numpy()
    data_np = (data_np > threshold).astype(np.float32)

    # Save as .npy file
    np.save(file_path, data_np)
    print(f"Rock data saved to {file_path}")


def rock_data_augmentation(data, conditions):
    """
    Rock data augmentation function

    Args:
        data: rock data tensor
        conditions: condition information

    Returns:
        tuple: (augmented data, augmented conditions)
    """
    augmented_data = data.clone()
    augmented_conditions = conditions.copy()

    # Random rotation
    if torch.rand(1) > 0.5:
        # 3DRandom rotation
        k = torch.randint(0, 4, (1,)).item()
        augmented_data = torch.rot90(augmented_data, k, dims=[2, 3])

    # Random flip
    if torch.rand(1) > 0.5:
        augmented_data = torch.flip(augmented_data, dims=[2])  # height flip

    if torch.rand(1) > 0.5:
        augmented_data = torch.flip(augmented_data, dims=[3])  # width flip

    if torch.rand(1) > 0.5:
        augmented_data = torch.flip(augmented_data, dims=[4])  # depth flip

    return augmented_data, augmented_conditions
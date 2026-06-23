"""
Device handling utilities for CPU/GPU compatibility.

Provides convenience functions matching PA3 patterns for device management.
"""

import torch


def get_device():
    """
    Get device (cuda if available, else cpu).

    Returns:
        torch.device: 'cuda' if GPU available, else 'cpu'

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"Using device: {device_name}")
    return device


def move_to_device(data, device):
    """
    Move data (tensor, dict, list, tuple) to device.

    Args:
        data: Tensor, dict of tensors, list/tuple of tensors, or nested structure
        device: torch.device

    Returns:
        Same structure with all tensors on device

    Example:
        >>> device = get_device()
        >>> batch = move_to_device(batch, device)  # Moves all tensors in dict
        >>> images = move_to_device(images, device)  # Single tensor
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(val, device) for key, val in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data

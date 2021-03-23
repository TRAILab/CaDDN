import torch


def normalize_coords(coords, shape):
    """
    Normalize coordinates of a grid between [-1, 1]
    Args:
        coords [torch.Tensor(..., 2)]: Coordinates in grid
        shape [torch.Tensor(2)]: Grid shape [H, W]
    Returns:
        norm_coords [torch.Tensor(.., 2)]: Normalized coordinates in grid
    """
    min_n = -1
    max_n = 1
    shape = torch.flip(shape, dims=[0])  # Reverse ordering of shape

    # Subtract 1 since pixel indexing from [0, shape - 1]
    norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
    return norm_coords
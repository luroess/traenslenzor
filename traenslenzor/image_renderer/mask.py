import numpy as np
from numpy._typing import NDArray


def create_box_mask(
    x: int, y: int, w: int, h: int, original_size: tuple[int, int]
) -> NDArray[np.uint8]:
    """
    Create a binary mask with ones in the specified box region.

    Parameters:
    -----------
    x : int
        X-coordinate of the top-left corner of the box
    y : int
        Y-coordinate of the top-left corner of the box
    w : int
        Width of the box
    h : int
        Height of the box
    original_size : tuple
        (height, width) of the output mask

    Returns:
    --------
    numpy.ndarray
        Binary mask with ones in the box region, zeros elsewhere
    """
    # Create a mask filled with zeros
    mask = np.zeros(original_size, dtype=np.uint8)

    # Set the box region to ones
    # Ensure we don't go out of bounds
    y_end = min(y + h, original_size[0])
    x_end = min(x + w, original_size[1])

    mask[y:y_end, x:x_end] = 1

    return mask

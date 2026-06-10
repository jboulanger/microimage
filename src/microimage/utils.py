from typing import List, Union

import numpy as np
import skimage as ski


def mask_to_yolo_contours(
    labels: np.ndarray, class_index: int = 0
) -> List[List[Union[int, float]]]:
    """Convert a label image or binary mask to YOLO contour lines using scikit-image.

    Parameters
    ----------
    labels : np.ndarray
        2D label image or binary mask.
    class_index : int
        The class index to assign to the contours. Defaults to 0.

    Returns
    -------
    List[List[Union[int, float]]]
    List of contours, where each contour is [class_index, x1, y1, x2, y2, ...].
    Coordinates are normalized to the image size [0, 1].
    """
    if labels.ndim != 2:
        return []

    h, w = labels.shape

    # Ensure disconnected components are treated as separate objects if input is binary
    unique_vals = np.unique(labels)
    if unique_vals.size == 2 and 0 in unique_vals:
        labeled_imgs = ski.measure.label(labels)
    else:
        labeled_imgs = labels

    props = ski.measure.regionprops(labeled_imgs)
    all_lines = []

    for prop in props:
        # Pad the object mask to prevent find_contours from clipping at edges
        mask = np.pad(prop.image_filled.astype(np.uint8), 2)

        # Extract contours (y, x coordinates)
        contours = ski.measure.find_contours(mask, level=0.5)
        if not contours:
            continue

        # Take the largest contour for this object (typically the outer boundary)
        largest_contour = max(contours, key=len)

        # Simplify the polygon to keep vertex count reasonable
        simplified = ski.measure.approximate_polygon(largest_contour, tolerance=1.0)

        min_y, min_x, _, _ = prop.bbox
        normalized_poly = []
        for y, x in simplified:
            # Adjust for padding and bounding box offset, then normalize
            global_x = (x - 2 + min_x) / w
            global_y = (y - 2 + min_y) / h
            # Round for storage efficiency and clamp to [0, 1]
            normalized_poly.append(float(np.clip(round(global_x, 6), 0, 1)))
            normalized_poly.append(float(np.clip(round(global_y, 6), 0, 1)))

        all_lines.append([class_index] + normalized_poly)

    return all_lines

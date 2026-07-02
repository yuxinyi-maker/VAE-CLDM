
from skimage import measure
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage import measure
from scipy.spatial import cKDTree
import warnings




def compute_max_diameter(region_labels):
    """
    Safely compute the maximum diameter of a 3D binary mask, fully handling various edge cases
    Args:
        region_labels: 3D numpy array representing a binary mask
    Returns:
        Maximum diameter (float), returns 0.0 for invalid cases
    """
    # Input validation
    if not isinstance(region_labels, np.ndarray) or region_labels.ndim != 3:
        warnings.warn("Input must be a 3D numpy array", RuntimeWarning)
        return 0.0

    # Get non-zero voxel coordinates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = np.argwhere(region_labels > 0)

    # Handle empty region
    if len(coords) == 0:
        return 0.0

    # Method selection: automatically choose based on data size
    if len(coords) <= 10000:  # Small data uses exact computation
        try:
            max_dist = 0.0
            for i in range(0, len(coords), 100):  # Process in chunks to avoid memory issues
                chunk = coords[i:i + 100]
                dists = np.linalg.norm(chunk[:, np.newaxis] - coords, axis=2)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    chunk_max = np.nanmax(dists)
                if chunk_max > max_dist:
                    max_dist = chunk_max
            return max_dist
        except MemoryError:
            pass  # Fall back to approximate method on memory error

    # Large data uses KD-tree approximation
    tree = cKDTree(coords)
    sample_size = min(1000, len(coords))
    max_dist = 0.0

    for _ in range(3):  # Multiple samplings to improve accuracy
        sample_indices = np.random.choice(len(coords), size=sample_size, replace=False)
        for i in sample_indices:
            dist, _ = tree.query(coords[i], k=2)  # Query nearest two points (including self)
            if len(dist) > 1 and not np.isnan(dist[1]):  # Skip self and NaN values
                max_dist = max(max_dist, dist[1])

    return max_dist * 1.05  # Add 5% compensation factor


def demotoo(a):
    vsb = []
    for data in a:
        labeled_data, num_features = measure.label(data, return_num=True)
        vs = []
        for i in range(1, num_features + 1):
            region_labels = labeled_data == i
            diameter = compute_max_diameter(region_labels)
            vs.append(diameter)
        vs = np.array(vs)
        idex = np.nonzero(vs)
        vb = vs[idex]
        tt = np.mean(vb)
        bb = np.std(vb)
        vsb.append([tt, bb])
    return np.array(vsb)


def demotoo1(data):
    """
    Compute pore size parameters
    Input data format: 0=pore, 1=solid (consistent with training data)
    """
    vsb = []
    
    # Note: measure.label labels non-zero regions
    # If input is 0=pore, we need to invert first, or directly label the 0-valued regions
    # Here we label the pore regions (values equal to 0)
    pore_mask = (data == 0).astype(np.uint8)  # Create pore mask
    labeled_data, num_features = measure.label(pore_mask, return_num=True)
    
    vs = []
    for i in range(1, num_features + 1):
        region_labels = labeled_data == i
        diameter = compute_max_diameter(region_labels)
        vs.append(diameter)
    vs = np.array(vs)
    idex = np.nonzero(vs)
    vb = vs[idex]
    tt = np.mean(vb)
    bb = np.std(vb)
    vsb.append([tt, bb])
    return np.array(vsb)

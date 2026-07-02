
import os
import numpy as np
from demoto import demotoo1

path = "/hy-tmp/VAEDDPM/111"
path_new = "/hy-tmp/VAEDDPM/val_data_limestone/"

# Create output directory
os.makedirs(path_new, exist_ok=True)

for filename in os.listdir(path):
    if not filename.endswith('.npy'):
        continue
    
    filepath = os.path.join(path, filename)
    
    # Load generated data (0=pore, 1=solid)
    data = np.load(filepath)
    
    # Compute pore size parameters (demotoo1 modified to support 0=pore format)
    label = np.squeeze(demotoo1(data))
    pore_size_mean = np.around(label[0], 3)
    pore_size_std = np.around(label[1], 3)
    
    # Save: keep original format (0=pore, 1=solid)
    output_filename = f"{pore_size_mean}_{pore_size_std}.npy"
    np.save(os.path.join(path_new, output_filename), data)
    
    print(f"Processed: {filename} -> {output_filename}")


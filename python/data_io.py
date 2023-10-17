import os
import numpy as np
import open3d as o3d

def load_as_array(path):
    # Loads the data from an .xyz file into a numpy array.
    # Also returns a boolean indicating whether per-point labels are available.
    data_array = np.loadtxt(data_dir + scans[0], comments='//') # raw data as np array, of shape (nx6) or (nx8) if labels are available.
    labels_available = data_array.shape[1] == 8
    return data_array, labels_available

def load_as_o3d_cloud(path):
    # Loads the data from an .xyz file into an open3d point cloud object.
    data, labels_available = load_as_array(path)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,0:3])
    pc.colors = o3d.utility.Vector3dVector(data[:,3:6]/255)
    labels = None
    if labels_available:
        labels = data[:,6:]
    return pc, labels_available, labels

def save_data_as_xyz(data, path):
    # To save your own data in the same format as we used, you can use this function.
    # Edit as needed with more or fewer columns.
    with open(path, 'w') as f:
        f.write("//X Y Z R G B class instance\n")
        np.savetxt(f, data, fmt=['%.6f', '%.6f', '%.6f','%d', '%d', '%d'])
    return


if __name__ == '__main__':
   
   '''Example usage of the functions defined in this file'''

    # Please specify where the data set is located:
    data_dir = '/home/.../'
    
    all_files = sorted(os.listdir(data_dir))
    scans = [fname for fname in all_files if fname.endswith(".xyz")]
    
    # Load the first scan in the data set for example:
    pointcloud, labels_available, labels = load_as_o3d_cloud(data_dir + scans[0])
    # Visualize the point cloud:
    o3d.visualization.draw_geometries([pointcloud])
"""
Data loader to filter an annotated xyz file by class into individual files
"""

import os
import numpy as np
import open3d as o3d


def load_as_array(path):
    # Loads the data from an .xyz file into a numpy array.
    # Also returns a boolean indicating whether per-point labels are available.
    data_array = np.loadtxt(
        path, comments="//"
    )  # raw data as np array, of shape (nx6) or (nx8) if labels are available.
    labels_available = data_array.shape[1] == 8
    return data_array, labels_available


def load_as_o3d_cloud(path):
    # Loads the data from an .xyz file into an open3d point cloud object.
    data, labels_available = load_as_array(path)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:, 0:3])
    pc.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)
    labels = None
    if labels_available:
        labels = data[:, 6:]
    return pc, labels_available, labels


def save_data_as_xyz(data, path):
    # To save your own data in the same format as we used, you can use this function.
    # Edit as needed with more or fewer columns.
    with open(path, "w") as f:
        f.write("//X Y Z R G B class instance\n")
        np.savetxt(f, data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])
    return


def filter_xyz(path, filename, selected_cats, filter_instances=False, return_ids=False):

    class_cats = {
        "leaf": 1,
        "petiole": 2,
        "berry": 3,
        "flower": 4,
        "crown": 5,
        "background": 6,
        "other": 7,
    }
    selected_cats = [class_cats[cat] for cat in selected_cats]

    with open(data_dir + filename + ".xyz") as f:
        lines = (line for line in f if not line.startswith("//X"))
        data = np.loadtxt(lines, skiprows=1)

    entries = []
    instances = []
    for entry in data:
        x, y, z, r, g, b, semantic_label, instance_label = entry

        if semantic_label in selected_cats:
            entries.append(entry)
            instances.append(instance_label)

    # Add function to filter out any disconnected clusters (?)

    if filter_instances:
        pcds = []
        instances = list(set(instances))
        points = [[]] * len(instances)
        inst_mapper = dict(zip(instances, np.arange(0, len(instances))))

        for x, y, z, r, g, b, semantic_label, instance_label in entries:

            if len(points[inst_mapper[instance_label]]) == 0:
                points[inst_mapper[instance_label]] = [[x, y, z]]
            else:
                points[inst_mapper[instance_label]].append([x, y, z])

        for inst in instances:
            inst_pts = np.array(points[inst_mapper[inst]])

            opcd = o3d.geometry.PointCloud()
            opcd.points = o3d.utility.Vector3dVector(inst_pts)
            pcds.append(opcd)
            # o3d.io.write_point_cloud(f"Data/{filename}/petiole_{inst}.ply",opcd)
        if return_ids:
            return pcds, np.array(instances).astype("int")
        return pcds

    else:
        points = []
        colours = []
        for x, y, z, r, g, b, semantic_label, instance_label in entries:
            points.append([x, y, z])
            colours.append([r, g, b])

        points = np.array(points)
        colours = np.array(colours) / 255

        opcd = o3d.geometry.PointCloud()
        opcd.points = o3d.utility.Vector3dVector(points)
        opcd.colors = o3d.utility.Vector3dVector(colours)
        # o3d.io.write_point_cloud(f"Data/{filename}/selected.ply",opcd)
        return opcd


if __name__ == "__main__":

    # Please specify where the data set is located:
    data_dir = "path/to/files/Zara/annotated/"

    # Save each stem instance as a separate ply
    out_dir = "path/to/files/" + "petiole_instances/"  #'stem_class' #
    return_ids = True

    # Filter out the background
    # out_dir = "path/to/files/plant_only"
    # return_ids = False

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_files = sorted(os.listdir(data_dir))
    scans = [fname for fname in all_files if fname.endswith(".xyz")]

    # Load the first scan in the data set for example:
    for scan in scans:
        pointcloud, labels_available, labels = load_as_o3d_cloud(data_dir + scan)

        # Visualize the point cloud:
        # o3d.visualization.draw_geometries([pointcloud])

        fname = scan[:-4]
        print(fname)

        if return_ids:

            selected_cats = ["petiole", "crown"]
            pcd = filter_xyz(data_dir, scan[:-4], selected_cats, filter_instances=False)
            o3d.io.write_point_cloud(
                f"{out_dir}/{fname}_stem.ply", pcd, write_ascii=True
            )

        else:
            selected_cats = ["petiole", "leaf", "berry", "flower", "crown"]
            pcd = filter_xyz(
                data_dir,
                scan[:-4],
                selected_cats,
                filter_instances=False,
                return_ids=False,
            )
            o3d.io.write_point_cloud(f"{out_dir}/{fname}.ply", pcd, write_ascii=True)

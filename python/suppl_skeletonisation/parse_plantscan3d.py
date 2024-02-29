"""
Script to load exported nodes from PlantScan3D and convert to O3D
"""

import open3d as o3d
import numpy as np
import os


sample = 'xu'    
base_path_in = f"path/to/files/petiole_instances/{sample}_raw"
base_path_out = f"path/to/files/petiole_instances/{sample}"

filenames = os.listdir(base_path_in)

for filename in filenames:
    if filename.endswith (".ply"): 
        continue

    print(filename)

    with open(f"{base_path_in}/{filename}") as file:
        lines = [line.rstrip() for line in file]

    edges = []
    nodes = []

    root = lines[3].split("\t")
    root_id = root[0]
    nodes.append([root[3],root[4],root[5]])

    radius=True
    if root[6] == "None":
        radius=False

    for i in range(4,len(lines)):    
        if radius:
            vid, parentid, edgetype, x, y, z, r = lines[i].split("\t")
        else:
            vid, parentid, edgetype, x, y, z = lines[i].split("\t")
        nodes.append([x,y,z])
        edges.append([int(vid), int(parentid)])

    nodes = np.array(nodes).reshape(-1,3)
    edges = np.array(edges) - int(root_id)

    # adjust edges
    dictionary = dict(zip(np.unique(edges), np.arange(0,len(nodes))))
    edges = [[dictionary[e0],dictionary[e1]] for e0,e1 in edges]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nodes)

    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(
                nodes), lines=o3d.utility.Vector2iVector(edges))

    if not os.path.exists(base_path_out):
        os.mkdir(base_path_out)
        
    o3d.io.write_line_set(
                f"{base_path_out}/{filename}.ply", line_set, write_ascii=True)
    print(f"{base_path_out}/{filename}.ply")

        
        
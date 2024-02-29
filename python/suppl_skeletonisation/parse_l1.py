from uu import Error
from cloudvolume import Skeleton
import numpy as np
import open3d as o3d
import os


def parse_skel(filename):
    result = {}
    lines = open(filename).readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != ""]

    assert lines[0].startswith("ON")
    num_original = int(lines[0].split()[1])
    lines = lines[1:]
    result["original"] = np.stack(
        [np.array([float(x) for x in line.split()]) for line in lines[:num_original]],
        axis=0,
    )
    lines = lines[num_original:]

    # NOTE: samples are potentially inf
    assert lines[0].startswith("SN")
    num_sampled = int(lines[0].split()[1])
    lines = lines[1:]
    result["sample"] = np.stack(
        [np.array([float(x) for x in line.split()]) for line in lines[:num_sampled]],
        axis=0,
    )
    lines = lines[num_sampled:]

    assert lines[0].startswith("CN")
    num_branches = int(lines[0].split()[1])
    lines = lines[1:]

    branches = []
    for _ in range(num_branches):
        assert lines[0].startswith("CNN")
        num_nodes = int(lines[0].split()[1])
        lines = lines[1:]
        branches.append(
            np.stack(
                [
                    np.array([float(x) for x in line.split()])
                    for line in lines[:num_nodes]
                ],
                axis=0,
            )
        )
        lines = lines[num_nodes:]
    result["branches"] = branches
    len_branches = [x.shape[0] for x in branches]
    
    if len(len_branches) == 0:
        print("No branches")
        return None
    
    assert lines[0] == "EN 0"
    lines = lines[1:]
    assert lines[0] == "BN 0"
    lines = lines[1:]

    assert lines[0].startswith("S_onedge")
    lines = lines[1:]
    result["sample_onedge"] = np.array(list(map(int, lines[0].split()))) > 0
    lines = lines[1:]

    assert lines[0].startswith("GroupID")
    lines = lines[1:]
    result["sample_groupid"] = np.array(list(map(int, lines[0].split())))
    lines = lines[1:]

    # flattened branches
    assert lines[0].startswith("SkelRadius")
    lines = lines[1:]
    result["branches_skelradius"] = np.split(
        np.array(list(map(float, lines[0].split()))), np.cumsum(len_branches)
    )[:-1]
    lines = lines[1:]

    assert lines[0].startswith("Confidence_Sigma")  
    lines = lines[1:]
    result["sample_confidence_sigma"] = np.array(list(map(float, lines[0].split())))
    lines = lines[1:]

    assert lines[0] == "SkelRadius2 0"
    lines = lines[1:]
    assert lines[0] == "Alpha 0"
    lines = lines[1:]

    assert lines[0].startswith("Sample_isVirtual")
    lines = lines[1:]
    result["sample_isvirtual"] = np.array(list(map(int, lines[0].split()))) > 0
    lines = lines[1:]

    assert lines[0].startswith("Sample_isBranch")
    lines = lines[1:]
    result["sample_isbranch"] = np.array(list(map(int, lines[0].split()))) > 0
    lines = lines[1:]

    assert lines[0].startswith("Sample_radius")
    lines = lines[2:]

    assert lines[0].startswith("Skel_isVirtual")
    lines = lines[1:]
    result["skel_isvirtual"] = np.split(
        np.array(list(map(int, lines[0].split()))) > 0, np.cumsum(len_branches)
    )[:-1]
    lines = lines[1:]

    # NOTE: this does not generate anything useful, as samples are potentially inf
    assert lines[0].startswith("Corresponding_sample_index")
    lines = lines[1:]
    result["corresponding_sample_index"] = np.split(
        np.array(list(map(int, lines[0].split()))), np.cumsum(len_branches)
    )[:-1]
    lines = lines[1:]

    assert len(lines) == 0

    return result

def to_cloud_volume_skeleton(parsed):
    branch_length = np.array([len(x) for x in parsed["branches"]])
    flattened_vertices = np.concatenate(parsed["branches"])
    flattened_radii = np.concatenate(parsed["branches_skelradius"])

    # NOTE: may need to replace this with np.isclose to allow within epsilon dists
    unique, index, inverse = np.unique(
        flattened_vertices, axis=0, return_index=True, return_inverse=True
    )
    edges = []

    branch_inverse = np.split(inverse, np.cumsum(branch_length))[:-1]
    for branch in branch_inverse:
        edges.append(np.stack([branch[:-1], branch[1:]], axis=1))
    edges = np.concatenate(edges, axis=0)

    flattened_radii = flattened_radii[np.argsort(inverse)]
    radii = np.split(flattened_radii, np.cumsum(branch_length))[:-1]
    assert max([len(np.unique(x)) for x in radii])
    radii = flattened_radii[index]

    skel = Skeleton(vertices=unique, edges=edges, radii=radii)

    return skel

def generate_skeleton(name):
    skel = parse_skel(f"Results/l1_medial/raw/{name}")
    print(skel)
    if skel is not None:
        skeleton = to_cloud_volume_skeleton(skel)   
        return skeleton
    else:
        return None

if __name__ == "__main__":  
    
    # filenames = os.listdir(f"Results/l1_medial/raw/")  
    filenames = ['katrina2_20220729_11.skel']
    # print(filenames)
    for filename in filenames:      
        if  filename[-4:] == '.ply' :
            continue #for reruns
        print('filename: ', filename)
        skeleton = generate_skeleton(filename)  
        print(skeleton)
        
        if skeleton is not None:
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(
                        skeleton.vertices), lines=o3d.utility.Vector2iVector(skeleton.edges))
            o3d.io.write_line_set(
                    f"Results/l1_medial/{filename[:-5]}.ply", line_set, write_ascii=True)
            # o3d.visualization.draw_geometries([line_set])

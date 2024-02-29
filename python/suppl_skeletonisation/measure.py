import open3d as o3d
import numpy as np
import os
import networkx as nx

from GraphMatching3D.graph_compare.graph_compare import * 

sample = 'xu'    
base_path_in = f"path/to/files/petiole_instances/{sample}"

filenames = os.listdir(base_path_in)

write_info = []
for filename in filenames:
    if not filename.endswith (".ply"): 
        continue

    # load data
    lineset = o3d.io.read_line_set(f"{base_path_in}/{filename}")
    nodes = np.array(lineset.points)
    edges = np.array(lineset.lines)
   
    adj_matrix = create_adjacency_matrix(edges,nodes)   
   
    graph = create_graph_from_adjacency_matrix(adj_matrix)   

    # estimate length - bear in mind the estimate might branch even though the gt doesn't
    all_paths = find_all_paths_between_leaf_nodes(
        graph
    )  # (edgepair) : [[edge0, edge1...]]
       
    end_pairs = all_paths.keys()  
    unique_edges = list(end_pairs)    
    est_dists = [sum_path(graph, adj_matrix, e0, e1) for e0, e1 in unique_edges] 

    # Save the best distance per file (timestep/sample) - longest? 
    i_max = np.argmax(est_dists)
    max_segment_path = all_paths[unique_edges[i_max]] 
    # print(max_segment_path) 
    max_segment_length = est_dists[i_max]    
    write_info.append([filename, max_segment_length])

np.savetxt(f'Results/{sample}_length.txt', write_info, fmt='%s')    
    
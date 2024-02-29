import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.optimize import linear_sum_assignment
from GraphMatching3D.graph_compare.graph_compare import create_adjacency_matrix, create_graph_from_adjacency_matrix, find_leaf_nodes
import random
random.seed(42)

def load_as_array(path):
    # Loads the data from an .xyz file into a numpy array.
    # Also returns a boolean indicating whether per-point labels are available.  
    data_array = np.loadtxt(path, comments='//') # raw data as np array, of shape (nx6) or (nx8) if labels are available.
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

def filter_xyz(path, filename, selected_cats, filter_instances = False, return_ids = False):  
          
    class_cats = {"leaf":1, "petiole":2, "berry":3, "flower":4, "crown":5, "background":6, "other":7}
    selected_cats = [class_cats[cat] for cat in selected_cats]
    
    with open(data_dir + filename + ".xyz") as f:
        lines = (line for line in f if not line.startswith("//X")) 
        data = np.loadtxt(lines,skiprows=1) 

    entries = []
    instances = []
    for entry in data:        
            x,y,z,r,g,b,semantic_label,instance_label = entry 

            if  semantic_label in selected_cats:     
                entries.append(entry)
                instances.append(instance_label)

    # Add function to filter out any disconnected clusters (?)

    if filter_instances:  
        pcds = []   
        instances = list(set(instances))     
        points = [[]] * len(instances)        
        inst_mapper = dict(zip(instances, np.arange(0,len(instances))))
            
        for x,y,z,r,g,b,semantic_label,instance_label in entries:
            
            if len(points[inst_mapper[instance_label]]) == 0:
                points[inst_mapper[instance_label]] = [[x,y,z]]
            else:                        
                points[inst_mapper[instance_label]].append([x,y,z])            
           
        for inst in instances:           
            inst_pts = np.array(points[inst_mapper[inst]])
          
            opcd = o3d.geometry.PointCloud()
            opcd.points = o3d.utility.Vector3dVector(inst_pts)
            pcds.append(opcd)
            # o3d.io.write_point_cloud(f"Data/{filename}/petiole_{inst}.ply",opcd) 
        if return_ids:
             return pcds, np.array(instances).astype('int') 
        return pcds                          

    else:    
        points = []
        for x,y,z,r,g,b,semantic_label,instance_label in entries:      
            points.append([x,y,z]) 
        
        points = np.array(points)

        opcd = o3d.geometry.PointCloud()
        opcd.points = o3d.utility.Vector3dVector(points)
        # o3d.io.write_point_cloud(f"Data/{filename}/selected.ply",opcd) 
        return opcd 

def generate_random_color(): 
            r = random.randint(0, 255) 
            g = random.randint(0, 255) 
            b = random.randint(0, 255) 
            return (r, g, b)          
  
def potential_stem_leaf_junctions(crown_centre, skel_files):
    stem_junctions = []   
    for i in range(0,len(skel_files)):
        ply = skel_files[i]
        nodes = np.asarray(ply.points)
        edges = np.asarray(ply.lines)
        adj_matrix = create_adjacency_matrix(edges,nodes)
        graph = create_graph_from_adjacency_matrix(adj_matrix)
        end_args  = find_leaf_nodes(graph)
        end_pts = nodes[end_args]
            
            # remove the one closer to the crown
        a = end_pts[0]
        b = end_pts[1]
        dist0 = np.linalg.norm(a-crown_centre)
        dist1 = np.linalg.norm(b-crown_centre)

        if dist0 < dist1: # a is closer to crown
            stem_junctions.append(b)
        else:
            stem_junctions.append(a)
    return stem_junctions

def dict_to_string(associations):
    string_associations = {}
    for k,v in associations.items():
        string_associations[str(k)] = str(v)
    return string_associations

if __name__ == '__main__':
   
    '''Example usage of the functions defined in this file'''

    # Please specify where the data set is located:
    data_dir = 'path/to/files/Katrina/annotated/'
    gt_skel_dir = "path/to/files/petiole_instances/gt"   
    output = "stem_associations"
    
    all_files = sorted(os.listdir(data_dir))
    scans = [fname for fname in all_files if fname.endswith(".xyz")]
    

    f = open("leaf_associations.json")
    all_plants = json.load(f)

    file_name_assoc = {}

    col_map = {}
    unique_stems = {} # one of these per scan family (zara vs katrina)# stem id: unique stem id

    g_assoc = {}
    old_corr = {}

    ply_to_id = {}
    
    # Load the first scan in the data set for example:
    for scan in scans:   
        if scan[:-4] not in all_plants.keys():
            print(f"{scan[:-4]} not associated yet")
            continue 

        print(scan)
        pointcloud, labels_available, labels = load_as_o3d_cloud(data_dir + scan)
        # Visualize the point cloud:
        class_labels = labels[:,0].astype('float')
        instance_labels = labels[:,1].astype('float') 
        
        # filter for leaf class        
        leaves, ids = filter_xyz(data_dir, scan[:-4], ['leaf'], filter_instances=True, return_ids = True)  
        stems, local_stem_ids = filter_xyz(data_dir, scan[:-4], ['petiole'], filter_instances=True, return_ids = True)  
              
        # continue
        crown = filter_xyz(data_dir, scan[:-4], ['crown'], filter_instances=False, return_ids = False)

        crown_centre = np.mean(np.asarray(crown.points),axis=0)  
        leaf_junctions =  [np.mean(np.asarray(leaf.points),axis=0)  for leaf in leaves]     
                 
        # ---- Update to global instance IDs                

        global_leaf_ids = []
        for i in ids:
            label_map = all_plants[scan[:-4]]  
            # print(label_map)           
            global_leaf_ids.append(label_map[str(i)]) 
        
        # populate col map
        combined = []
        for i in global_leaf_ids:
            if i not in col_map.keys():
                col_map[i] = generate_random_color()
                 
        # associate stems and leaves
        # load stem gt skels
               
        skel_files = []
        for i in range(0,len(local_stem_ids)): #  because stem ids were saved like this in data_loader
            path = f"{gt_skel_dir}/{scan[:-4]}_{i}.ply"
            ply = o3d.io.read_line_set(path)   
            skel_files.append(ply)      
              
              
            ply_to_id[f"{scan[:-4]}_{i}"] = local_stem_ids[i]

        print(np.arange(0,len(local_stem_ids)))
        print(local_stem_ids)
        
        # get end pts
        stem_junctions = potential_stem_leaf_junctions(crown_centre, skel_files)

        n_stems = len(stem_junctions)
        n_leaves = len(leaf_junctions)
        cost_matrix = np.zeros((n_stems,n_leaves))  

        for i in range(0,n_stems):
            for j in range(0,n_leaves):
                cost_matrix[i][j] = np.linalg.norm(stem_junctions[i] - leaf_junctions[j])       
          
        # assign least cost end pt to the leaf   
        s_ind, l_ind = linear_sum_assignment(cost_matrix, maximize=False) # these are just indicies NOT ids
        
        # LOCAL stem identified:  unique stem_id (across scans) :      
        for i in local_stem_ids:
            i = int(i)
            if len(unique_stems) == 0:
                new_val = 0
            else:
                new_val = max(unique_stems.values()) + 1 
            # unique_stems[new_key] = 
            unique_stems[f"{scan[:-4]}_{i}"] = new_val 
        # unique stems: {'katrina2_20220512_3': 0, 'katrina2_20220512_6': 1}   

        this_scan = {}
        for i in local_stem_ids:
            # if i not in this_scan.keys():
            this_scan[str(i)] = -1           
                              
        correspondence = {} # global stem id: global leaf ----- HERE trying to flip the logic so it works
        
        for i in range(0,len(s_ind)):         
            local_id = local_stem_ids[s_ind[i]] # eg 3 -> map to zero
            unique_global_s = unique_stems[f"{scan[:-4]}_{local_id}"]
            correspondence[unique_global_s] = global_leaf_ids[l_ind[i]]  

            this_scan[f"{local_id}"] = global_leaf_ids[l_ind[i]] 

        # print(correspondence)
        for k,v in correspondence.items():
            if v not in old_corr.values():
                if len(g_assoc) == 0:
                    g_assoc[k] = 0 
                else:
                    vals = g_assoc.values()
                    new_gid = max(vals)+1         
                    g_assoc[k] = new_gid
            else: # we have seen this global id before                
                matching_key = list(old_corr.keys())[list(old_corr.values()).index(v)]
                g_id = g_assoc[matching_key]
                g_assoc[k] = g_id
        
        old_corr = correspondence
               
        # this_scan = {}      

        
        file_name_assoc[scan[:-4]] = dict_to_string(this_scan) # local to global ids
               
        stem_col_map = {}
        for i in local_stem_ids:            
            g_id = unique_stems[f"{scan[:-4]}_{i}"]     
            if g_id in correspondence.keys():                                     
                stem_col_map[g_id] = col_map[correspondence[g_id]]
            else:                    
                stem_col_map[g_id] = generate_random_color()           

        # ---- recolour each leaf according to id     
        for i in range(0,len(leaves)): #                   
            colours = np.ones((len(np.array(leaves[i].points)),3)) * col_map[global_leaf_ids[i]]   
            colours = np.array(colours)/255  
            leaves[i].colors = o3d.utility.Vector3dVector(np.asarray(colours))               

        for i in range(0,len(stems)): #    
            colours = np.ones((len(np.array(stems[i].points)),3)) * stem_col_map[unique_stems[f"{scan[:-4]}_{local_stem_ids[i]}"]]   
            colours = np.array(colours)/255 
            stems[i].colors = o3d.utility.Vector3dVector(np.asarray(colours))  
        
        organs = leaves.copy()
        organs.extend(stems)
        # print(associations)
        
        # o3d.visualization.draw_geometries([*organs,*skel_files])  

    # write to json for stems: original ids : global ids
    # string_associations = dict_to_string(associations)

    # print('final')
    # print(associations)
    with open(f"{output}.json", 'w') as f:
        json.dump(file_name_assoc, f)

    with open(f"ply_to_local.json","w") as f:
        json.dump(dict_to_string(ply_to_id),f)
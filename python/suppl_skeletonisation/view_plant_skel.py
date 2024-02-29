import open3d as o3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data_dir = 'path/to/files/Katrina/annotated'
method = 'none'
plant_name = 'katrina2_20220729'

def plot_skel(ax, lineset, color='red', s=0.5, label=None):
    nodes = np.array(lineset.points)
    edges = np.array(lineset.lines) 

    ax.scatter([X[0] for X in nodes], [X[1] for X in nodes], [X[2]
                for X in nodes], alpha=1, color=color, s=s, label='skeleton points')
    
    for edge in edges:
        n0 = nodes[edge[0]]
        n1 = nodes[edge[1]]
        x = [n0[0],n1[0]]
        y = [n0[1],n1[1]]
        z = [n0[2],n1[2]]
        ax.plot3D(x,y,z, color = color)


filenames = os.listdir("path/to/files/petiole_instances/gt/")

# filter by plant name
filenames = [s for s in filenames if plant_name in s]
# filenames = ['katrina2_20220729_11']

original = o3d.io.read_point_cloud(f"{data_dir}/{plant_name}.xyz") ###
# original.paint_uniform_color([0,0,0])
points = np.array(original.points)
indices = np.random.choice(len(points), 40000)
points = points[indices]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection="3d")  

ax.scatter([X[0] for X in points], [X[1] for X in points], [X[2]
                    for X in points], alpha=1, color='grey', s=0.1, label='raw points')

special_filenames = ["katrina2_20220729_2","katrina2_20220729_19","katrina2_20220729_20"]

for filename in filenames:
    if filename[-4:] == ".ply":   
        color = 'green'
        
        if filename[:-4] == special_filenames[0]: # L1 fails 11,24
            color = 'lime'
        elif filename[:-4] == special_filenames[1]: # weird split
            color = 'cyan'
        elif filename[:-4] == special_filenames[2]: # decent performance
            color = 'hotpink'
        
        # k_cyl = o3d.io.read_line_set(f"Results/kcylinder/{sample}/{filename}" )#.asc", format="xyzrgb")
        # k_cyl.paint_uniform_color([1, 0, 0])
        if method == 'l1':
            lineset = o3d.io.read_line_set(f"Results/l1_medial/{filename}")
        # l1.paint_uniform_color([0, 1, 0])
        elif method == 'xu':
            lineset = o3d.io.read_line_set(f"path/to/files/petiole_instances/xu/{filename}")      
        # xu.paint_uniform_color([0, 0, 1])
        elif method == 'som':
            lineset = o3d.io.read_line_set(f"Results/som/{filename}")
        else:
            lineset = None
        # som.paint_uniform_color([0, 1, 1])
        
        gt = o3d.io.read_line_set(f"path/to/files/petiole_instances/gt/{filename}")#.asc", format="xyzrgb")
        # # gt.paint_uniform_color([0, 1, 0])
        # print(f"Data/young/{sample}/skel/{filename}")

        # o3d.visualization.draw_geometries([original, som]) # k_cyl, l1, xu, 
        
        plot_skel(ax, gt, color=color, s=0.1, label=None)   
        if lineset is not None:
            plot_skel(ax, lineset, color='red', s=0.1, label=None) 

   

# # ax.legend()
ax.view_init(elev=16, azim=131)
# ax.set_xlim([-68, -63])  # Adjust the X-axis limits
# ax.set_ylim([90,100])  # Adjust the Y-axis limits
# ax.set_zlim([178,183])   # Adjust the Z-axis limits
# # # # # # # Hide grid lines
ax.grid(False)
plt.axis('off')

plt.show()
# plt.savefig(f"Results/images/circle/{plant_name}_allstems.png", bbox_inches='tight', dpi=300)



colors = ['lime','cyan','hotpink']
downsample = []


def plot_stem(filename, color):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d") 

    stem = o3d.io.read_point_cloud(f"path/to/files/petiole_instances/{filename}.ply")
    gt = o3d.io.read_line_set(f"path/to/files/petiole_instances/gt/{filename}.ply")
    
    points = np.array(stem.points)
    print(len(points))
    indices = np.random.choice(len(points), 955)
    points = points[indices]

    plot_skel(ax, gt, color=color, s=0.1, label=None)
    ax.scatter([X[0] for X in points], [X[1] for X in points], [X[2]
                        for X in points], alpha=1, color='grey', s=0.1, label='raw points')
    ax.grid(False)
    plt.axis('off')

    plt.show()
    # plt.savefig(f"Results/images/circle/{filename}.png", bbox_inches='tight', dpi=300)


for i,f in enumerate(special_filenames):
    plot_stem(f,colors[i])
    
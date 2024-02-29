import open3d as o3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

method = 'xu'
filename = 'katrina2_20220729_24.ply'

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

if filename[-4:] == ".ply":   
    print(filename[:-4])

    original = o3d.io.read_point_cloud(f"path/to/files/petiole_instances/{filename}") ###
    # original.paint_uniform_color([0,0,0])
    points = np.array(original.points)

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
    # som.paint_uniform_color([0, 1, 1])
    
    gt = o3d.io.read_line_set(f"path/to/files/petiole_instances/gt/{filename}")#.asc", format="xyzrgb")
    # # gt.paint_uniform_color([0, 1, 0])
    # print(f"Data/young/{sample}/skel/{filename}")

    # o3d.visualization.draw_geometries([original, som]) # k_cyl, l1, xu,   
         
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection="3d")        

    ax.scatter([X[0] for X in points], [X[1] for X in points], [X[2]
                for X in points], alpha=1, color='grey', s=0.1, label='raw points')

    plot_skel(ax, gt, color='green', s=0.1, label=None)   
    plot_skel(ax, lineset, color='red', s=0.1, label=None) 

   

# ax.legend()
# ax.view_init(elev=-1, azim=-180)
# ax.set_xlim([-68, -63])  # Adjust the X-axis limits
# ax.set_ylim([90,100])  # Adjust the Y-axis limits
# ax.set_zlim([178,183])   # Adjust the Z-axis limits
# # # # # # # Hide grid lines
ax.grid(False)
plt.axis('off')

# plt.show()
plt.savefig(f"Results/images/{filename[:-4]}_{method}.png", bbox_inches='tight', dpi=300)


# 'katrina2_20220531_3.ply'
# ax.view_init(elev=-46, azim=-45)
# ax.set_xlim([-50, 10])  # Adjust the X-axis limits
# ax.set_ylim([375, 445])  # Adjust the Y-axis limits
# ax.set_zlim([120,220])   # Adjust the Z-axis limits

# katrina2_20220729_19
# ax.view_init(elev=82, azim=-138)
# ax.set_xlim([-200, -130])  # Adjust the X-axis limits
# ax.set_ylim([120, 200])  # Adjust the Y-axis limits
# ax.set_zlim([110,210]) 

# katrina2_20220729_2
# ax.view_init(elev=93, azim=-135)
# ax.set_xlim([-100, 20])  # Adjust the X-axis limits
# ax.set_ylim([200, 270])  # Adjust the Y-axis limits
# ax.set_zlim([110,150]) 

# katrina2_20220608_3
# x.view_init(elev=-35, azim=-135)
# ax.set_xlim([-20, 20])  # Adjust the X-axis limits
# ax.set_ylim([140,200])  # Adjust the Y-axis limits
# ax.set_zlim([140,210])
    



    
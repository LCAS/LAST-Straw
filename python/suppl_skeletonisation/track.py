import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def dict_to_int(associations):
    int_assoc = {}
    for k, v in associations.items():
        int_assoc[int(k)] = int(v)
    return int_assoc


def load_df():
    f = open("stem_associations.json")
    associations = json.load(f)  # stem unique id -> global stem id

    f2 = open("ply_to_local.json")
    ply_associations = json.load(f2)

    df = pd.read_csv("Results/xu_length.txt", sep=" ", names=["name", "length"])

    df["scan"] = (
        df["name"].str.split("_", n=2, expand=True).iloc[:, 0]
        + "_"
        + df["name"].str.split("_", n=2, expand=True).iloc[:, 1]
    )
    df["name"] = (
        df["name"].str.split(".", n=2, expand=True).iloc[:, 0]
    )  # the name saved on file (ply name)
    df["plant_name"] = df["name"].str.split("_", n=2, expand=True).iloc[:, 0]
    df["date"] = df["name"].str.split("_", n=2, expand=True).iloc[:, 1]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # Not associated yet
    df = df[df["plant_name"] == "katrina2"]
    df = df[df["scan"] != "katrina2_20220721"]
    df = df[df["scan"] != "katrina2_20220729"]

    def map_id(row):
        # map name + ply instance to stem id to global id
        local_id = ply_associations[row["name"]]
        global_id = associations[row["scan"]][str(local_id)]
        return global_id

    df["instance"] = df.apply(map_id, axis=1)

    df = df[df["instance"] != "-1"]

    df["length"] = df["length"].astype("float")
    return df


def plot_length_over_time(instance_df):
    import datetime

    # plt.gcf().set_size_inches(10, 6)
    fig = plt.figure()
    for name, instance in instance_df:
        if not all(instance["plant_name"].values == "katrina2"):
            continue

        if len(instance) < 2:
            continue

        instance.sort_values(by=["date"], inplace=True)
        plt.plot(
            instance["date"],
            instance["length"],
            label=name,
            marker="o",
            linewidth=2,
            markersize=8,
        )

    # Adding labels and legend
    plt.title(f"Stem length for plant A2 over time")
    plt.xlabel("Scan date")
    plt.ylabel("Length [mm]")

    all_scan_dates_A2 = ["0512", "0519", "0525", "0531", "0608"]
    datetime_str = [
        "05-12-2022",
        "05-19-2022",
        "05-25-2022",
        "05-31-2022",
        "06-08-2022",
    ]
    datet = [
        datetime.datetime.strptime(date_string, "%m-%d-%Y")
        for date_string in datetime_str
    ]

    labels = ["12.05", "19.05", "25.05", "31.05", "08.06"]

    plt.xticks(datet, labels=labels, rotation=45, ha="right")

    # plt.tight_layout()
    plt.legend(title="Instance", loc="lower right", ncol=1)
    plt.savefig(f"Results/images/Time-series/tracked_stem.png", dpi=300)
    plt.show()


def calculate_principal_axes(point_cloud):
    covariance_matrix = np.cov(point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx]


def plot_points(ax, points, i, color="grey", label=None):
    principal_axes1 = calculate_principal_axes(points)

    ax[i].scatter(
        np.dot(points, principal_axes1)[:, 0],
        np.dot(points, principal_axes1)[:, 1],
        np.dot(points, principal_axes1)[:, 2],
        c=color,
        marker=".",
    )

    ax[i].grid(False)
    ax[i].axis("off")
    ax[i].set_title(label, fontsize=10)


# Helper function to set the same scale for all axes
def set_axes_aspect_equal(ax):
    """Set equal aspect ratio for all axes."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    x_middle = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]
    y_middle = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_instance_over_time(instance_df, data_dir):

    for name, instance in instance_df:
        instance.sort_values(by=["date"], inplace=True)

        # look up what the filename is of the saved stem
        ply_names = instance["name"].values
        dates = instance["date"]

        dates = dates.dt.strftime("%Y-%m-%d")
        dates = dates.values
        inst = instance["instance"].values[0]
        lengths = instance["length"].values

        if len(ply_names) == 1:
            continue

        fig, axs = plt.subplots(1, len(ply_names), subplot_kw=dict(projection="3d"))

        # load that file
        for i, ply_name in enumerate(ply_names):
            print(ply_name)
            stem_pcd = o3d.io.read_point_cloud(f"{data_dir}/{ply_name}.ply")
            points = np.asarray(stem_pcd.points)

            # Translate the point cloud to the origin
            centroid = np.mean(points, axis=0)

            points = points - centroid

            # o3d.visualization.draw_geometries([stem_pcd])
            name = "{}\nLength: {:.2f}".format(dates[i], lengths[i])

            plot_points(axs, points, i, label=name)

        def zoom(ax, factor):
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            x_center = np.mean(x_limits)
            y_center = np.mean(y_limits)
            z_center = np.mean(z_limits)
            x_range = x_limits[1] - x_limits[0]
            y_range = y_limits[1] - y_limits[0]
            z_range = z_limits[1] - z_limits[0]
            ax.set_xlim3d([x_center - x_range * factor, x_center + x_range * factor])
            ax.set_ylim3d([y_center - y_range * factor, y_center + y_range * factor])
            ax.set_zlim3d([z_center - z_range * factor, z_center + z_range * factor])

        for ax in axs:
            zoom(ax, 0.2)
            set_axes_aspect_equal(ax)
            # Set azimuth angle to 0 for each subplot
            ax.view_init(azim=0)

        plt.tight_layout()

        # Save the figure
        plt.savefig(f"Results/images/Time-series/{inst}.png", dpi=300)
        # plt.show()


if __name__ == "__main__":
    data_dir = "path/to/files/petiole_instances/"

    df = load_df()

    instance_df = df.groupby("instance")

    plot_length_over_time(instance_df)

    plot_instance_over_time(instance_df, data_dir)

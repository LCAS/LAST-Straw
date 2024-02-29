"""
Script to assess the skeletonisation results
"""

import open3d as o3d
import numpy as np
import os

from GraphMatching3D.graph_compare.graph_compare import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_points(ax, Xs, alpha=0.1, color="b", label="", s=30, marker="."):
    # Plot the data points
    ax.scatter(
        [X[0] for X in Xs],
        [X[1] for X in Xs],
        [X[2] for X in Xs],
        alpha=alpha,
        color=color,
        label=label,
        s=s,
        marker=marker,
    )


method = "l1"
gt_path_in = f"path/to/files/petiole_instances/gt"
if method == "xu":
    est_path_in = f"path/to/files/petiole_instances/xu"
elif method == "som":
    est_path_in = "path/to/files/Results/som"
elif method == "l1":
    est_path_in = "path/to/files/Results/l1_medial"


filenames = os.listdir(gt_path_in)

write_info = []
for filename in filenames:
    if not filename.endswith(".ply"):
        continue

    print(filename)

    # load data - gt
    gt_lineset = o3d.io.read_line_set(f"{gt_path_in}/{filename}")
    gt_nodes = np.array(gt_lineset.points)
    gt_edges = np.array(gt_lineset.lines)

    # gt_adj_matrix = create_adjacency_matrix(gt_edges,gt_nodes)
    # gt_graph = create_graph_from_adjacency_matrix(gt_adj_matrix)

    # load data -
    # check if file exists
    if not os.path.isfile(f"{est_path_in}/{filename}"):
        print(f"No skel for {filename}")
        write_info.append([filename, 0, 0, 0, 0, 0, 0, 0])
    else:
        est_lineset = o3d.io.read_line_set(f"{est_path_in}/{filename}")
        est_nodes = np.array(est_lineset.points)
        est_edges = np.array(est_lineset.lines)

        # xu_adj_matrix = create_adjacency_matrix(xu_edges,gt_nodes)
        # xu_graph = create_graph_from_adjacency_matrix(xu_adj_matrix)

        # generate dense graph
        dense_s = 0.3

        est_dense_nodes, est_dense_edges, est_adj_matrix = create_dense(
            est_nodes, est_edges, s=dense_s, return_adj=True
        )

        gt_dense_nodes, gt_dense_edges, gt_adj_matrix = create_dense(
            gt_nodes, gt_edges, s=dense_s, return_adj=True
        )

        gt_graph = create_graph_from_adjacency_matrix(gt_adj_matrix)
        est_graph = create_graph_from_adjacency_matrix(est_adj_matrix)

        display = False
        if display:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            plot_points(ax, est_dense_nodes, color="orange", alpha=1)
            plot_points(ax, gt_dense_nodes, color="purple", alpha=1)

            plt.show()

        ### compute assessment metrics

        # PARAMETERS
        match_thresh = 0.396
        t_line = 0.396

        match_dict, tp_e = match_polyline_graphs(
            gt_graph, est_graph, gt_dense_nodes, est_dense_nodes, match_thresh, t_line
        )

        tp, fn, _ = confusion_matrix(match_dict, est_graph)  #
        fp = list(set(est_graph.keys()) - set(tp_e))

        est_tp_matches = corresponding_tp(match_dict)

        assert len(tp) == len(est_tp_matches)
        assert len(tp) + len(fn) == len(
            gt_graph.keys()
        ), f"{len(tp) + len(fn)},{len(gt_graph.keys())}"

        # Report metrics
        # ---precision and recall
        display = False
        false_positives = est_dense_nodes[fp]
        true_positives_e = est_dense_nodes[est_tp_matches]
        true_positives = gt_dense_nodes[tp]
        false_negatives = gt_dense_nodes[fn]

        if len(true_positives) + len(false_positives) > 0:
            report_precision = len(true_positives) / (
                len(true_positives) + len(false_positives)
            )  # how many of the positives are real
        else:
            report_precision = np.nan

        report_recall = len(true_positives) / (
            len(true_positives) + len(false_negatives)
        )  # how many of the actual positives were detected
        if report_precision + report_recall == 0:
            report_f1 = -1
        else:
            report_f1 = (
                2
                * report_precision
                * report_recall
                / (report_precision + report_recall)
            )

        if display:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            plot_points(ax, true_positives_e, color="lime", alpha=1)
            plot_points(ax, false_positives, color="blue", alpha=1)
            plot_points(ax, true_positives, color="green", alpha=1)
            plot_points(ax, false_negatives, color="red", alpha=1)
            plt.show()

        # -- total matched length

        node_list = np.unique(est_dense_edges.flatten())
        node_list = np.setdiff1d(node_list, fp)  # remove any unmatched nodes

        contiguous = find_contiguous_sections(est_graph, node_list)

        gt_ends_i = find_leaf_nodes(gt_graph)

        percentage_len_matched = 0
        if len(contiguous) > 0:
            len_contig = [
                sum_path(est_graph, est_adj_matrix, x[0], x[-1]) for x in contiguous
            ]
            m = np.argmax(
                len_contig
            )  # do we want this to be the most matched nodes or longest matching seg - latter

            gt_dist = sum_path(gt_graph, gt_adj_matrix, gt_ends_i[0], gt_ends_i[1])

            # calc_len = sum_path(est_graph, est_adj_matrix, contiguous[m][0], contiguous[m][-1])
            report_contig = len_contig[m] / gt_dist  # the longest matched segment / gt
            contiguous_tp = est_dense_nodes[contiguous[m]]

            display = False

            total_matching_len = np.sum(len_contig)  # should be len MATCHED contiguous

            percentage_len_matched = total_matching_len / gt_dist

            # diff in measured length (error)
            all_paths = find_all_paths_between_leaf_nodes(
                est_graph
            )  # (edgepair) : [[edge0, edge1...]]

            end_pairs = all_paths.keys()
            unique_edges = list(end_pairs)
            est_dists = [
                sum_path(est_graph, est_adj_matrix, e0, e1) for e0, e1 in unique_edges
            ]

            i_max = np.argmax(est_dists)
            max_segment_path = all_paths[unique_edges[i_max]]
            max_segment_length = est_dists[i_max]
            absolute_percentage_error = abs(gt_dist - max_segment_length) / gt_dist

        # -- n branches and n segments
        est_node_list = np.unique(est_dense_edges.flatten())
        contiguous = find_contiguous_sections(est_graph, est_node_list)
        n_segments = len(contiguous)

        # # determine how many end points there are
        est_ends = find_leaf_nodes(est_graph)
        n_ends = len(est_ends)

        info = [
            filename,
            report_precision,
            report_recall,
            report_f1,
            percentage_len_matched,
            absolute_percentage_error,
            n_ends,
            n_segments,
        ]

        write_info.append(info)

np.savetxt(f"Results/{method}_assessment.txt", write_info, fmt="%s")

# print results with stdev for reporting

write_info = np.array(write_info)

p = write_info[:, 1].astype("float")
r = write_info[:, 2].astype("float")
f1 = write_info[:, 3].astype("float")
plm = write_info[:, 4].astype("float")
abs_e = write_info[:, 5].astype("float")
e = write_info[:, 6].astype("int")
s = write_info[:, 7].astype("int")

print(
    "{:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f}".format(
        np.nanmean(p),
        np.nanstd(p),
        np.nanmean(r),
        np.nanstd(r),
        np.nanmean(f1),
        np.nanstd(f1),
        np.nanmean(plm),
        np.nanstd(plm),
        np.nanmean(abs_e),
        np.nanstd(abs_e),
        np.nanmean(e),
        np.nanstd(e),
        np.nanmean(s),
        np.nanstd(s),
    )
)

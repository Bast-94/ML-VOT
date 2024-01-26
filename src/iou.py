from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BoundingBox = namedtuple("BoundingBox", ("bb_left", "bb_top", "bb_width", "bb_height"))


def print_bounding_box(bb):
    print(
        f"bb_left: {bb.bb_left}, bb_top: {bb.bb_top}, bb_width: {bb.bb_width}, bb_height: {bb.bb_height}"
    )


def intersection_box(bb1, bb2):
    bb_left = max(bb1.bb_left, bb2.bb_left)
    bb_top = max(bb1.bb_top, bb2.bb_top)
    bb_right = min(bb1.bb_left + bb1.bb_width, bb2.bb_left + bb2.bb_width)
    bb_bottom = min(bb1.bb_top + bb1.bb_height, bb2.bb_top + bb2.bb_height)
    height = bb_bottom - bb_top
    width = bb_right - bb_left
    if height < 0 or width < 0:
        return BoundingBox(bb_left, bb_top, 0, 0)

    return BoundingBox(bb_left, bb_top, bb_right - bb_left, bb_bottom - bb_top)


def iou(bb1, bb2):
    intersection = intersection_box(bb1, bb2)
    intersection_area = intersection.bb_width * intersection.bb_height
    bb1_area = bb1.bb_width * bb1.bb_height
    bb2_area = bb2.bb_width * bb2.bb_height
    union_area = bb1_area + bb2_area - intersection_area
    return intersection_area / union_area


def plot_bounding_boxes(bb1: BoundingBox, bb2: BoundingBox, ax: plt.Axes):
    """
    Plots two bounding boxes
    :param bb1: bounding box 1
    :param bb2: bounding box 2
    :return: None
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.set_xlim(
        min(bb1.bb_left, bb2.bb_left) - 10,
        max(bb1.bb_left + bb1.bb_width, bb2.bb_left + bb2.bb_width) + 10,
    )
    ax.set_ylim(
        min(bb1.bb_top, bb2.bb_top) - 10,
        max(bb1.bb_top + bb1.bb_height, bb2.bb_top + bb2.bb_height) + 10,
    )
    ax.add_patch(
        Rectangle(
            (bb1.bb_left, bb1.bb_top),
            bb1.bb_width,
            bb1.bb_height,
            fill=False,
            color="red",
        )
    )
    ax.add_patch(
        Rectangle(
            (bb2.bb_left, bb2.bb_top),
            bb2.bb_width,
            bb2.bb_height,
            fill=False,
            color="blue",
        )
    )
    iou_score = iou(bb1, bb2)
    bb_intersection = intersection_box(bb1, bb2)
    ax.add_patch(
        Rectangle(
            (bb_intersection.bb_left, bb_intersection.bb_top),
            bb_intersection.bb_width,
            bb_intersection.bb_height,
            fill=True,
            color="green",
        )
    )
    ax.set_title(f"IoU: {iou_score:.2f}")


def similarity_matrix_iou(bb_list1: list[BoundingBox], bb_list2: list[BoundingBox]):
    """
    Computes the similarity matrix between two lists of bounding boxes
    :param bb_list1: list of bounding boxes
    :param bb_list2: list of bounding boxes
    :return: similarity matrix
    """
    sim_matrix = np.zeros((len(bb_list1), len(bb_list2)))
    for i, bb1 in enumerate(bb_list1):
        for j, bb2 in enumerate(bb_list2):
            sim_matrix[i, j] = iou(bb1, bb2)
    return sim_matrix


if __name__ == "__main__":
    fig, ax = plt.subplots()

    # create simple line plot

    # add rectangle to plot

    fig, ax = plt.subplots()
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])

    bb1 = BoundingBox(0, 0, 10, 10)
    print("bb1:")
    print_bounding_box(bb1)
    ax.add_patch(
        Rectangle(
            (bb1.bb_left, bb1.bb_top),
            bb1.bb_width,
            bb1.bb_height,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )
    bb2 = BoundingBox(5, 5, 10, 10)
    print("bb2:")
    print_bounding_box(bb2)
    ax.add_patch(
        Rectangle(
            (bb2.bb_left, bb2.bb_top),
            bb2.bb_width,
            bb2.bb_height,
            fill=False,
            edgecolor="blue",
            linewidth=2,
        )
    )
    bb3 = intersection_box(bb1, bb2)
    ax.add_patch(
        Rectangle(
            (bb3.bb_left, bb3.bb_top),
            bb3.bb_width,
            bb3.bb_height,
            fill=True,
            linewidth=2,
            color="green",
        )
    )
    print("bb3:")
    print_bounding_box(bb3)
    print("iou(bb1, bb2):", iou(bb1, bb2))

    fig.savefig("iou.png")

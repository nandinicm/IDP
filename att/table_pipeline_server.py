from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
import sys

from table_inference.lib.model.config import cfg
from table_inference.lib.utils.nms import non_max_suppression

import tensorflow as tf
import numpy as np
import os, cv2
import re
from table_inference.lib.nets.vgg16 import vgg16
from table_inference.tools.demo import parse_args, demo
from table_inference.Table_cell_data import getout_table_cells_information

import json
from ocr_pattern_hypothesis.utils.frame_utils import calculate_all_points, draw_polygon


def ui_format(table_json):
    ui_data = []
    for table_id, data in table_json.items():
        table_temp = {}
        table_temp["id"] = str(table_id)
        table_temp["confidenceScore"] = str(100)
        table_temp["tag"] = str(data["table"]["tags"])
        table_temp["styles"] = {}
        table_temp["type"] = "table"
        table_temp["coordinates"] = {"x": str(data["table"]["coord"][1]),
                                     "y": str(data["table"]["coord"][0]),
                                     "width": str(data["table"]["coord"][3] - data["table"]["coord"][1]),
                                     "height": str(data["table"]["coord"][2] - data["table"]["coord"][0])}
        table_temp["tableCols"] = []
        table_temp["tableRows"] = []

        for col_data in data["columns"]:
            col_temp = {}
            col_temp["id"] = str(col_data["index"])
            col_temp["tags"] = str(col_data["tags"])
            col_temp["coordinates"] = {"x": str(col_data["coord"][1]),
                                       "y": str(col_data["coord"][0]),
                                       "width": str(col_data["coord"][3] - col_data["coord"][1]),
                                       "height": str(col_data["coord"][2] - col_data["coord"][0])}
            col_temp["styles"] = {}
            col_temp["type"] = "col"
            table_temp["tableCols"].append(col_temp)

        for row_data in data["rows"]:
            temp_row = {}
            temp_row["id"] = str(row_data["index"])
            temp_row["isHeader"] = "true" if row_data["index"] == 0 else "false"
            temp_row["coordinates"] = {"x": str(row_data["coord"][1]),
                                       "y": str(row_data["coord"][0]),
                                       "width": str(row_data["coord"][3] - row_data["coord"][1]),
                                       "height": str(row_data["coord"][2] - row_data["coord"][0])}
            temp_row["styles"] = {}
            temp_row["type"] = "row"
            temp_row["cells"] = []
            temp_row["tags"] = str(row_data["tags"])
            for cell_data in data["data"]:
                if str(cell_data["row_index"]) == str(row_data["index"]):
                    temp_cell = {}
                    temp_cell["id"] = str(cell_data["col_index"])
                    temp_cell["isNull"] = "false"
                    temp_cell["colSpan"] = "1"
                    temp_cell["rowSpan"] = "1"
                    temp_cell["value"] = str(cell_data["value"])
                    temp_cell["styles"] = {}
                    temp_cell["type"] = "cell"
                    temp_cell["coordinates"] = {"x": str(cell_data["coord"][1]),
                                                "y": str(cell_data["coord"][0]),
                                                "width": str(cell_data["coord"][3] - cell_data["coord"][1]),
                                                "height": str(cell_data["coord"][2] - cell_data["coord"][0])}
                    temp_row["cells"].append(temp_cell)
            table_temp["tableRows"].append(temp_row)
        ui_data.append(table_temp)
    return {"tables": ui_data}


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    #
    # determine the coordinates of the intersection rectangle
    x_left = int(max(bb1['x1'], bb2['x1']))
    y_top = int(max(bb1['y1'], bb2['y1']))
    x_right = int(min(bb1['x2'], bb2['x2']))
    y_bottom = int(min(bb1['y2'], bb2['y2']))

    if x_right < x_left or y_bottom < y_top:
        return 0.0, 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    total_area = float(bb1_area + bb2_area - intersection_area)
    # iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # assert iou >= 0.0
    # assert iou <= 1.0
    return total_area, intersection_area


def get_tables(table_json):
    tab_coords = []
    for table in table_json["patches"]:
        tab_coords.append([table["y1"], table["x1"], table["y2"], table["x2"]])
    return tab_coords


if __name__ == '__main__':

    base_folder = sys.argv[1]
    image_folder = base_folder + '/images/'
    tfmodel = '/home/devops/models_phase_2_107/res101_faster_rcnn_iter_107000.ckpt'

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    tables_json_folder = base_folder + "/tables/"
    if not os.path.isdir(tables_json_folder):
        os.mkdir(tables_json_folder)

    # model path
    # demonet = args.demo_net
    # dataset = args.dataset

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    net = vgg16()

    net.create_architecture("TEST", 3,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    for file in list(set(os.listdir(image_folder))):
        if file.startswith("."):
            continue
        try:
            image = cv2.imread(image_folder + file)
        except FileNotFoundError:
            print("image not found")
            continue
        print(image_folder + file)
        # im=cv2.imread(image_folder+file)
        cls_boxes, cls_scores = demo(sess, net, image)
        new_boxes = [[int(c[0]), int(c[1]), int(c[2]), int(c[3])] for e, c in enumerate(cls_boxes) if
                     cls_scores[e] >= 0.9]
        DL_boxes = non_max_suppression(np.asarray(new_boxes))

        listed_tables = []
        for box in DL_boxes:
            points = calculate_all_points(((box[0], box[1]), (box[2], box[3]), 0))
            draw_polygon(image, points, (0,0,255),thickness=4)
            listed_tables.append([box[1], box[0], box[3], box[2]])
        evidence = json.load(open(base_folder + '/words/' + re.sub('.jpg$', '.json', file)))

        all_table_cell_info, table_json = getout_table_cells_information(listed_tables, image, evidence)
        table_file = re.sub('.jpg$', '.json', file)
        with open(tables_json_folder + table_file, "w") as tbfile:
            json.dump(ui_format(table_json), tbfile)
        for k, v in all_table_cell_info.items():
            for k1, v1 in v.items():
                k1 = [int(i) for i in k1.replace(" ", "")[1:-1].split(",")]
                cv2.rectangle(image, (k1[1], k1[0]), (k1[3], k1[2]), (0, 0, 0), thickness=2)

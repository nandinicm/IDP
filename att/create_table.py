import sys
import cv2
import numpy as np
from numpy import vectorize
import time
from IDP_pipeline.hypotheses.ocr_pattern_hypothesis.RND.Connected_islands import Connected_component_islands
import os
import json
from collections import Counter
from collections import OrderedDict
# import matplotlib.pyplot as plt
from pdf2image import convert_from_path

import ast
import json
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from ocr_pattern_hypothesis.utils import frame_utils
from ocr_pattern_hypothesis.frames.basic_frames import Word
import random
import pandas as pd
# STRUCTURE FRAMES
from ocr_pattern_hypothesis.frames.structure.engine import StructureEngine
from ocr_pattern_hypothesis.frames.structure.text import TextLine
from pymongo import MongoClient

mongo_ip = "mongodb://localhost"
client_name = "tangoe1"


def create_checking_image(img, evidence):
    new_img = np.zeros_like(img)
    # for k, v in evidence['page_0']['evidence_words'].items():
    #     c = v["assembled_result"][0]
    #     word=v["assembled_result"][1]
    #     cv2.putText(new_img, word, (c[1], c[0]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=.7, color=(255, 255, 255), thickness=3)

    cv2.imwrite("/home/amandubey/Videos/created_IMG.jpg", new_img)


def get_histogram(data, orientation):
    return data.shape[orientation] - np.count_nonzero(data, orientation)


def get_absolute_distance_between_bloks(self, rect1, rect2):  # (x1, y1, x1b, y1b), (x2, y2, x2b, y2b)):
    left = rect2[3] - rect1[1]  # always -ve
    right = rect2[1] - rect1[3]  # always +ve
    bottom = rect2[0] - rect1[2]  # always +ve
    top = rect2[2] - rect1[0]  # always -ve
    if top < 0:
        if left < 0:
            return math.sqrt(math.pow(top, 2) + math.pow(left, 2))
        elif right > 0:
            return math.sqrt(math.pow(top, 2) + math.pow(right, 2))
        else:
            return abs(top)
    elif bottom > 0:
        if left < 0:
            return math.sqrt(math.pow(bottom, 2) + math.pow(left, 2))
        elif right > 0:
            return math.sqrt(math.pow(bottom, 2) + math.pow(right, 2))
        else:
            return abs(bottom)
    elif left < 0:
        return abs(left)
    elif right > 0:
        return abs(right)
    else:  # rectangles intersect
        return 0


def is_inside(rect1, rect2):  # (x1, y1, x1b, y1b), (x2, y2, x2b, y2b)):
    # if check_data():
    left = rect2[3] < rect1[1]  # always -ve
    right = rect2[1] > rect1[3]  # always +ve
    bottom = rect2[0] > rect1[2]  # always +ve
    top = rect2[2] < rect1[0]  # always -ve
    if top or left or right or bottom:
        return False
    else:  # rectangles intersect
        return True


def find_rows_and_columns(table_array, table_borders):
    row_seperating_list = []
    col_seperating_list = []

    zero_counts = get_histogram(table_array, 1)
    max_value = max(zero_counts)
    line_flag = False
    for enum, row in enumerate(zero_counts):
        if row == table_array.shape[1] and not line_flag:

            row_seperating_list.append(table_borders[0] + enum)
            line_flag = True
        elif row < table_array.shape[1] and line_flag:
            line_flag = False

    zero_counts = get_histogram(table_array, 0)
    max_value = max(zero_counts)
    line_flag = False
    for enum, col in enumerate(zero_counts):
        if col == table_array.shape[0] and not line_flag:
            col_seperating_list.append(table_borders[1] + enum)
            line_flag = True
        elif col < table_array.shape[0] and line_flag:
            line_flag = False

    border_dict = {"rows": row_seperating_list, "cols": col_seperating_list}
    return border_dict


# def get_textlines(evidence, image):
#     s_engine = StructureEngine((
#         TextLine.generate,
#     ))
#     word_patches_dict = {}
#     structures = []
#     for each_evidence in evidence['page_0']['words']:
#         label = str(each_evidence['label'])
#         coordinates = (each_evidence['coordinates']['y'], each_evidence['coordinates']['x'],
#                        (each_evidence['coordinates']['height'] + each_evidence['coordinates']['y']),
#                        (each_evidence['coordinates']['width'] + each_evidence['coordinates']['x']))
#         label_word = label
#         word_patches_dict[coordinates] = label_word
#
#     try:
#         structures = s_engine.run(image, word_args=(word_patches_dict,))
#     except IndexError:
#         structures = []
#     structures = structures.filter(TextLine)
#     return structures


def get_textlines(evidence, image):
    s_engine = StructureEngine((
        TextLine.generate,
    ))
    word_patches_dict = {}
    structures = []
    for each_evidence in evidence['words']:
        label_word = str(each_evidence['label'])
        coordinates = (each_evidence['coordinates']['y'], each_evidence['coordinates']['x'],
                       (each_evidence['coordinates']['height'] + each_evidence['coordinates']['y']),
                       (each_evidence['coordinates']['width'] + each_evidence['coordinates']['x']))

        # xx=re.findall(r'[a-zA-Z0-9-+=-_]+', label_word)

        # if len(xx)<1:
        #     label_word=" "

        word_patches_dict[coordinates] = label_word

    try:
        structures = s_engine.run(image, word_args=(word_patches_dict,))
    except IndexError:
        structures = []
    structures = structures.filter(TextLine)
    return structures


"""
get rows of present in table
"""


def find_rows_of_table(table_array, table_borders):
    row_seperating_list = []

    zero_counts = get_histogram(table_array, 1)
    line_flag = False
    max_value = int(.97 * table_array.shape[1])
    for enum, row in enumerate(zero_counts):
        if row > max_value and not line_flag:
            if len(row_seperating_list) == 0 and enum != 0:
                row_seperating_list.append(table_borders[0])

            row_seperating_list.append(table_borders[0] + enum)
            line_flag = True
        elif row < max_value and line_flag:
            line_flag = False
    if not line_flag:
        row_seperating_list.append(table_borders[2])
    return row_seperating_list


def check_for_last_column(table_array):
    last_index = table_array.shape[1] - 1
    while True:
        if len(set(table_array[:, last_index])) > 1:
            return last_index + 1
        elif last_index < 0:
            print("Error in chech_for_last_column")
            break
        else:
            last_index = last_index - 1


"""
find coordinates for column
"""


def get_col_value(table_array: np.ndarray):
    zero_counts = list(get_histogram(table_array, 0))
    max_value = max(zero_counts)
    last_max_pos = table_array.shape[1] - 1
    first_max_pos = 0
    while True:
        if zero_counts[first_max_pos] == max_value:
            return first_max_pos
        elif first_max_pos > table_array.shape[1] - 1:
            break

        else:
            first_max_pos = first_max_pos + 1
    return 0  # while True:  #     if zero_counts[last_max_pos]==max_value:  #         return last_max_pos  #     elif last_max_pos<0:  #         break  #  #     else:  #         last_max_pos=last_max_pos-1  # return 0


""""
Removing column if 2 or more created for same seperation
"""


def clean_columns_list(table_array, col_seperating_list, table_borders):
    new_col_seperating_list = []
    previous_set = set()
    col_seperating_list.append(table_borders[3])

    for col_index in range(len(col_seperating_list) - 1):
        start_col = col_seperating_list[col_index] - table_borders[1]
        end_col = col_seperating_list[col_index + 1] - table_borders[1]

        latest_set = set(table_array[:, start_col:end_col].flat)
        check_previous_set = previous_set.difference(previous_set.intersection(latest_set))
        if len(previous_set) > 1:
            if len(latest_set.difference(previous_set)) > 0 and (len(check_previous_set)):
                previous_set = latest_set
                new_col_seperating_list.append(col_seperating_list[col_index])

        elif len(new_col_seperating_list) == 0:
            previous_set = latest_set
            new_col_seperating_list.append(col_seperating_list[col_index])
        elif len(latest_set) > len(previous_set):
            previous_set = latest_set
    new_col_seperating_list.append(col_seperating_list[-1])
    return new_col_seperating_list


"""
Running a filter for white and black pixels to get all columns
"""


def find_cols_of_table_using_filter(table_array, table_borders):
    mask_actual_width = 30
    col_seperating_threshold = .4
    mask = np.zeros((table_borders[2] - table_borders[0], mask_actual_width), dtype=np.int16)
    last_col_textlines = [0]
    col_seperating_list = []
    end_flag = True
    x1 = 0
    x2 = mask_actual_width
    mask_width = x2 - x1
    while end_flag:

        mask[:, 0:mask_width] = table_array[:, x1:x2]
        if mask_width < mask_actual_width:
            mask[:, mask_width:(mask.shape[1] - 1)] = 0
        previous_textlines = list(set(mask.flat).intersection(set(last_col_textlines)))
        # print(previous_textlines)
        new_textlines = list(set(mask.flat).difference(previous_textlines))

        if len(new_textlines) > 0:
            last_col_textlines = list(set(mask.flat))

        if (len(new_textlines) / len(previous_textlines)) > col_seperating_threshold:
            if x1 > 0:
                strt = x1 - mask_actual_width
                add_val = -mask_actual_width
            else:
                strt = x1
                add_val = 0
            get_col = get_col_value(table_array[:, strt:x2])
            col_seperating_list.append((get_col + x1 + table_borders[1] + add_val))

        x1 = x2
        if x1 == (table_array.shape[1] - 1):
            end_flag = False
        x2 = x2 + mask_actual_width
        if (x2) > (table_array.shape[1] - 1):
            x2 = (table_array.shape[1] - 1)
        mask_width = x2 - x1
    # print("COL BEFORE :",col_seperating_list)
    col_seperating_list = clean_columns_list(table_array, col_seperating_list, table_borders)
    # print("Columns :",col_seperating_list)
    return col_seperating_list


def find_cols_of_table_using_filter_001(table_array, table_borders):
    mask_actual_width = 30
    mask = np.zeros((table_borders[2] - table_borders[0], mask_actual_width), dtype=np.int16)
    last_col_values = [-1]
    col_seperating_list = []
    end_flag = True
    x1 = 0
    x2 = mask_actual_width
    mask_width = x2 - x1
    while end_flag:

        mask[:, 0:mask_width] = table_array[:, x1:x2]
        if mask_width < mask_actual_width:
            mask[:, mask_width:(mask.shape[1] - 1)] = 0
        latest_list = list(set(mask.flat).difference(last_col_values))
        reverse_list = list(set(last_col_values).difference(set(mask.flat)))
        if len(latest_list) > 0 and (len(reverse_list) != 0 or len(last_col_values) == 1):
            last_col_values = list(set(mask.flat))
            if x1 > 0:
                strt = x1 - mask_actual_width
                add_val = -mask_actual_width
            else:
                strt = x1
                add_val = 0
            get_col = get_col_value(table_array[:, strt:x2])
            col_seperating_list.append((get_col + x1 + table_borders[1] + add_val))
        x1 = x2
        if x1 == (table_array.shape[1] - 1):
            end_flag = False
        x2 = x2 + mask_actual_width
        if (x2) > (table_array.shape[1] - 1):
            x2 = (table_array.shape[1] - 1)
        mask_width = x2 - x1
    last_index = (check_for_last_column(table_array)) + table_borders[1]
    print("COL BEFORE :", col_seperating_list)
    if col_seperating_list[(len(col_seperating_list) - 1)] < last_index:
        col_seperating_list.append(last_index)
    print("Columns :", col_seperating_list)
    print()
    return col_seperating_list


def find_cols_of_table(table_array, table_borders):
    col_seperating_list = []
    zero_counts = get_histogram(table_array, 0)
    x, y = [], []
    for enum, val in enumerate(zero_counts):
        x.append(enum)
        y.append(val)
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    x_smooth = np.linspace(x_arr.min(), x_arr.max(), 50)
    y_smooth = spline(x_arr, y_arr, x_smooth)
    # plt.plot(x_smooth,y_smooth)
    # plt.plot(x,y)
    # minimas = (np.diff(np.sign(np.diff(y_smooth))) > 0).nonzero()[0] + 1
    maximas = (np.diff(np.sign(np.diff(y_smooth))) < 0).nonzero()[0] + 1
    max_list = []
    for i in maximas:
        max_list.append(y_smooth[i])
    value_flag = True
    max_list.sort()
    itteration = 0
    print("MAX LIST :", set(max_list))
    while value_flag:
        max_check = max_list[itteration]
        frac = (max_check / (table_borders[2] - table_borders[0]))
        itteration = itteration + 1
        print(frac)
        if frac > 0.60 or itteration == len(max_list):
            value_flag = False

    start_flag = False
    end_flag = False
    space_count = 0
    pre_val = 0
    for enum, col in enumerate(zero_counts):
        if col > max_check - 1:
            space_count = space_count + 1
            start_flag = True
        elif col < max_check - 1 and start_flag:
            end_flag = True
            start_flag = False
        if not start_flag and len(col_seperating_list) == 0:
            col_seperating_list.append(table_borders[1])
        if end_flag:
            end_flag = False
            sub = int(space_count / 2)
            if sub > 5:
                col_seperating_list.append(table_borders[1] + enum - sub)
            space_count = 0

        col_seperating_list.append(table_borders[3])

    # print(start_flag,"cols",border_dict["cols"])
    #
    # f, axarr = plt.subplots(2)
    #
    # axarr[0].plot( x,y)
    #
    # axarr[0].set_title('normal')
    # axarr[ 1].plot(x_smooth, y_smooth)
    # axarr[ 1].set_title('Smooth 40')
    #
    # plt.show()
    return col_seperating_list


"""
get cell coordinates using border_dict info
"""


def get_cell_coordinates(border_dict):
    cell_coor = []
    col_len = len(border_dict["cols"])
    row_len = len(border_dict["rows"])
    for row_no in range(1, row_len):

        for col_no in range(1, col_len):
            cell_coor.append((border_dict["rows"][row_no - 1], border_dict["cols"][col_no - 1],
                              border_dict["rows"][row_no], border_dict["cols"][col_no]))  # row_ui_data.append(temp_row)
    return cell_coor


"""
get words inside each cell
"""


def get_cell_words(textline_words_list, cell_coor):
    textline_words_list.sort()
    cell_coor.sort()
    word_enum = 0
    cell_words = {}
    enum_list = []

    for coor in cell_coor:
        cell_words[(coor)] = []
        if len(enum_list) > 0:
            new_textline_words_list = []
            enum_list.sort(reverse=True)
            for del_enum in enum_list:
                textline_words_list.pop(
                    del_enum)  # for new_enum in range(word_enum,len(textline_words_list)):  #     if new_enum not in enum_list:  #         new_textline_words_list.append(textline_words_list[new_enum])  # textline_words_list=new_textline_words_list
        enum_list = []
        textline_end = len(textline_words_list)

        for enum in range(word_enum, textline_end):
            if is_inside(coor, textline_words_list[enum][0]):
                cell_words[(coor)].append(textline_words_list[enum][1])
                enum_list.append(enum)

    return cell_words


def get_row_column_and_cell_level_info(border_dict, table_borders, cell_words, has_row_header=False,
                                       has_col_header=False, first_element_is_part_of=1):
    for coor, words in cell_words.items():
        word_string = ""
        for word in words:
            word_string = word_string + " " + word
        cell_words[coor] = word_string
    cols = border_dict["cols"]
    rows = border_dict["rows"]
    all_rows = []
    all_cols = []
    for enum in range(0, (len(rows) - 1)):
        all_rows.append([rows[enum], cols[0], rows[enum + 1], cols[-1]])
    for enum in range(0, (len(cols) - 1)):
        all_cols.append([rows[0], cols[enum], rows[-1], cols[enum + 1]])

    col_header_coord = []
    col_headers_data = []

    if has_col_header:
        col_header_coord = all_rows[0]
        all_rows.pop(0)
        for enum in range(0, (len(cols) - 1)):
            insert_value = cell_words[rows[0], cols[enum], rows[1], cols[enum + 1]]
            if len(insert_value) < 1:
                insert_value = None
            col_headers_data.append(insert_value)

    row_header_coord = []
    row_headers_data = []
    top_of_row_header = True
    if has_col_header:
        row_header_start_point = 1
    else:
        row_header_start_point = 0
    if has_row_header:
        row_header_coord = all_cols[0]
        all_cols.pop(0)
        for enum in range(row_header_start_point, (len(rows) - 1)):
            insert_value = cell_words[rows[enum], cols[0], rows[enum + 1], cols[1]]
            if len(insert_value) < 1:
                insert_value = None
            row_headers_data.append(insert_value)

            if top_of_row_header:
                top_of_row_header = False
                row_header_coord[0] = rows[enum]

    # if not has_col_header and not has_row_header:
    #     first_element_is_part_of=0
    # elif not has_col_header and has_row_header:
    #     first_element_is_part_of=2
    # elif has_col_header and not has_row_header:
    #     first_element_is_part_of=1
    #
    # if first_element_is_part_of==1 and has_row_header:
    #     row_headers_data.pop(0)
    #     row_header_coord[1]=cols[1]
    # if first_element_is_part_of==2 and has_col_header:
    #     col_headers_data.pop(0)
    #     col_header_coord[0]=rows[1]

    row_wise_data = []
    col_wise_data = []
    for each_enum, each_col in enumerate(all_cols):
        col_wise_data.append(
            {"index": each_enum, "tags": "", "rows": [col_enum for col_enum in range(0, len(all_rows))],
             "coord": each_col, "value": ""})
    for each_enum, each_row in enumerate(all_rows):
        row_wise_data.append(
            {"index": each_enum, "tags": "", "columns": [row_enum for row_enum in range(0, len(all_cols))],
             "coord": each_row, "value": ""})

    cell_wise_data = []
    for col_enum, each_col in enumerate(all_cols):
        for row_enum, each_row in enumerate(all_rows):
            cell_wise_data.append({"col_index": col_enum, "row_index": row_enum, "node_index": None, "tags": "",
                                   "value": cell_words[
                                       all_rows[row_enum][0], all_cols[col_enum][1], all_rows[row_enum][2],
                                       all_cols[col_enum][3]],
                                   "coord": [all_rows[row_enum][0], all_cols[col_enum][1], all_rows[row_enum][2],
                                             all_cols[col_enum][3]]})

    print()
    return row_wise_data, col_wise_data, col_headers_data, col_header_coord, row_headers_data, row_header_coord, cell_wise_data


"""
writes json format structure
"""


def create_json_structure(table_number, table_borders, border_dict, cell_words, table_json):
    row_wise_data, col_wise_data, col_headers_data, col_header_coord, row_headers_data, row_header_coord, cell_wise_data = get_row_column_and_cell_level_info(
        border_dict, table_borders, cell_words)

    table_json[table_number] = {"table": {"value": "", "tags": "", "coord": tuple(table_borders)},
                                "rows": row_wise_data, "columns": col_wise_data,
                                "column_headers_data": col_headers_data,
                                "column_headers_coord": col_header_coord, "row_headers_data": row_headers_data,
                                "row_headers_coord": row_header_coord,
                                "textNodes": [{"index": 0, "coord": [0, 0, 0, 0], "tags": ""}],
                                "data": cell_wise_data}


def ui_format(table_json, lbl):
    ui_data = []
    print("table json data", table_json)
    for table_id, data in table_json.items():

        # print("&&&&&&&&&&&&&&&&&&&&&&&")

        table_temp = {}
        table_temp["_id"] = str(table_id)
        table_temp["confidenceScore"] = str(100)
        table_temp["tag"] = str(data["table"]["tags"])
        table_temp["styles"] = {}
        table_temp["type"] = "table"
        table_temp["label"] = lbl

        table_temp["coordinates"] = {"x": str(data["table"]["coord"][1]), "y": str(data["table"]["coord"][0]),
                                     "width": str(data["table"]["coord"][3] - data["table"]["coord"][1]),
                                     "height": str(data["table"]["coord"][2] - data["table"]["coord"][0])}
        table_temp["tableCols"] = []
        table_temp["tableRows"] = []
        # print(table_temp)

        for col_data in data["columns"]:
            col_temp = {}
            col_temp["_id"] = str(col_data["index"])
            col_temp["tags"] = str(col_data["tags"])
            col_temp["coordinates"] = {"x": str(col_data["coord"][1]), "y": str(col_data["coord"][0]),
                                       "width": str(col_data["coord"][3] - col_data["coord"][1]),
                                       "height": str(col_data["coord"][2] - col_data["coord"][0])}
            col_temp["styles"] = {}
            col_temp["type"] = "col"
            table_temp["tableCols"].append(col_temp)

        for row_data in data["rows"]:
            temp_row = {}
            temp_row["_id"] = str(row_data["index"])
            temp_row["isHeader"] = "true" if row_data["index"] == 0 else "false"
            temp_row["coordinates"] = {"x": str(row_data["coord"][1]), "y": str(row_data["coord"][0]),
                                       "width": str(row_data["coord"][3] - row_data["coord"][1]),
                                       "height": str(row_data["coord"][2] - row_data["coord"][0])}
            temp_row["styles"] = {}
            temp_row["type"] = "row"
            temp_row["cells"] = []
            temp_row["tags"] = str(row_data["tags"])
            for cell_data in data["data"]:

                if str(cell_data["row_index"]) == str(row_data["index"]):
                    temp_cell = {}
                    temp_cell["_id"] = str(cell_data["col_index"])
                    temp_cell["isNull"] = "false"
                    temp_cell["colSpan"] = "1"
                    temp_cell["rowSpan"] = "1"
                    temp_cell["value"] = str(cell_data["value"])
                    temp_cell["styles"] = {}
                    temp_cell["type"] = "cell"
                    temp_cell["coordinates"] = {"x": str(cell_data["coord"][1]), "y": str(cell_data["coord"][0]),
                                                "width": str(cell_data["coord"][3] - cell_data["coord"][1]),
                                                "height": str(cell_data["coord"][2] - cell_data["coord"][0])}
                    temp_row["cells"].append(temp_cell)
                    print("INDEX ID :", temp_cell["_id"])
            table_temp["tableRows"].append(
                temp_row)  # print("final")  # print(table_temp)  # print(json.dumps(table_temp, separators=(',', ':')))  # json.dump(table_temp,"test_json.json")
        ui_data.append(table_temp)  # print(table_temp)
    # print('#'*200)
    # print(ui_data)
    # print(ui_data)
    # with open("UI_table_added_evidences.json", "w") as out_file:
    # json.dump(ui_data, out_file)
    # exit()
    print("showing ui data :", type(ui_data), ui_data)
    return ui_data[0]


"""
get all information
"""


def getout_table_cells_information(listed_tables, img, evidence):
    all_table_cell_info = {}
    structure = get_textlines(evidence, img)
    border_dict = {}
    c = [0, 0, 0, 0]
    table_json = {}

    for table_number, table_borders in enumerate(listed_tables):
        textline_words_list = []
        table_array = np.zeros(((table_borders[2] - table_borders[00]), (table_borders[3] - table_borders[1])),
                               dtype=np.int16)
        print("\n\nTABLE DATA : ", (table_array.shape), table_borders)
        for enum, textline in enumerate(structure):
            c[0] = (textline.coordinates[0][1])
            c[1] = (textline.coordinates[0][0])
            c[2] = (textline.coordinates[1][1])
            c[3] = (textline.coordinates[1][0])
            cv2.rectangle(img, (c[1], c[0]), (c[3], c[2]), (0, 0, 0), 3)  #####  Show Text line

            if is_inside(c, table_borders):
                p1, p2, p3, p4 = (c[0] - table_borders[0]), (c[2] - table_borders[0]), (c[1] - table_borders[1]), (
                        c[3] - table_borders[1])
                if p1 < 0:
                    p1 = 0
                if p2 > table_array.shape[0] - 1:
                    p2 = table_array.shape[0] - 1
                if p3 < 0:
                    p3 = 0
                if p4 > table_array.shape[1] - 1:
                    p4 = table_array.shape[1] - 1
                table_array[p1:p2, p3:p4] = enum + 1
                for itme in (textline.contains["words"]):
                    word_coor = (
                        itme.coordinates[0][1], itme.coordinates[0][0], itme.coordinates[1][1], itme.coordinates[1][0])
                    textline_words_list.append([word_coor, str(itme)])

        border_dict["rows"] = find_rows_of_table(table_array, table_borders)
        print("ROWS :", border_dict["rows"])

        border_dict["cols"] = find_cols_of_table_using_filter(table_array, table_borders)
        print("Columns :", border_dict["cols"])

        cell_coor = get_cell_coordinates(border_dict)
        print("cell_coor : ", cell_coor)
        for one_cell in cell_coor:
            cv2.rectangle(img, (one_cell[1], one_cell[0]), (one_cell[3], one_cell[2]),
                          (random.randint(1, 254), random.randint(1, 254), random.randint(1, 254)), thickness=2)

        cell_words = get_cell_words(textline_words_list, cell_coor)
        # print(cell_words)
        # exit()
        create_json_structure(table_number, table_borders, border_dict, cell_words, table_json)
        temp = {}
        for k, v in cell_words.items():
            temp[str(k)] = v

        all_table_cell_info[str(tuple(table_borders))] = temp
        evidence["tables"] = table_json
    print("sending back all info and table_json")
    return all_table_cell_info, table_json


# def create_from_given_data(structure, cell_coor):
#     c = [0, 0, 0, 0]
#     word_data = ""
#     textline_words_list = []
#     for enum, textline in enumerate(structure):
#         c[0] = (textline.coordinates[0][1])
#         c[1] = (textline.coordinates[0][0])
#         c[2] = (textline.coordinates[1][1])
#         c[3] = (textline.coordinates[1][0])
#         # cv2.rectangle(img, (c[1], c[0]), (c[3], c[2]), (0, 0, 0), 3)  #####  Show Text line
#
#         if is_inside(c, cell_coor):
#             for itme in (textline.contains["words"]):
#                 if is_inside([itme.coordinates[0][1], itme.coordinates[0][0], itme.coordinates[1][1],
#                               itme.coordinates[1][0]], cell_coor):
#                     word_data += str(itme) + ' '
#     return word_data


def intersection_area_coord(rect1, rect2):
    t = max((rect1[0], rect2[0]))
    l = max((rect1[1], rect2[1]))
    b = min((rect1[2], rect2[2]))
    r = min((rect1[3], rect2[3]))
    return [t, l, b, r]


def area_of_patch(rect):
    return ((rect[2] - rect[0]) * (rect[3] - rect[1]))


def create_from_given_data(structure, cell_coor):
    c = [0, 0, 0, 0]
    word_data = ""
    textline_words_list = []
    for enum, textline in enumerate(structure):
        c[0] = (textline.coordinates[0][1])
        c[1] = (textline.coordinates[0][0])
        c[2] = (textline.coordinates[1][1])
        c[3] = (textline.coordinates[1][0])
        # cv2.rectangle(img, (c[1], c[0]), (c[3], c[2]), (0, 0, 0), 3)  #####  Show Text line

        if is_inside(c, cell_coor):
            for itme in (textline.contains["words"]):
                if (area_of_patch(
                        intersection_area_coord([itme.coordinates[0][1], itme.coordinates[0][0], itme.coordinates[1][1],
                                                 itme.coordinates[1][0]], cell_coor)) / area_of_patch(
                    [itme.coordinates[0][1], itme.coordinates[0][0], itme.coordinates[1][1],
                     itme.coordinates[1][0]])) > .5:
                    word_data += str(itme) + ' '
    return word_data


def get_filename():
    localtime = time.asctime(time.localtime(time.time()))
    ltime = localtime.split(" ")
    time_stamp = ltime[2] + '_' + ltime[1] + '_' + ltime[4] + '_' + ltime[3] + '_'
    return time_stamp


if __name__ == '__main__':
    # uploadpath = sys.argv[1]
    # document_id = uploadpath.split('/')[-1]
    # pageId='5c5d550e6f23730e1f81de53'
    client = MongoClient(mongo_ip)
    db = client[client_name]

    pageId = sys.argv[1]
    documentId = sys.argv[2]

    # documentId= '5c5d541a6f23730e1f81de4f'
    all_data = list(db.fields.find({"documentId": documentId, "pageId": pageId}))
    page_name = all_data[0]['path'].split('jpg')[0]
    # print(page_name)
    img_name = os.getenv("HOME") + "/IDP/results/" + documentId + "/images/" + page_name + 'jpg'
    # img_name = "/home/amandubey/Downloads/5c5d541a6f23730e1f81de4f/Kone_waste_management_KCI009506.pdf_12_0.jpg"
    image = cv2.imread(img_name)

    # with open(uploadpath + "pages/" + f) as ev:
    with open(os.getenv("HOME") + "/IDP/results/" + documentId + "/words/" + page_name + 'json') as ev:
        evidence = ast.literal_eval(ev.read())
    # print("DOc id:", documentId)
    # print("Page id:", pageId)
    # print('IMAGE  :', image.shape)
    # print("EVIDENCES :", evidence)
    # print("ALL DATA ", all_data[0])
    new_created_tables = []
    try:

        # print(len(all_data[0]),type(all_data[0]),all_data[0])
        # for k,v in all_data[0].items():
        #     print(k,v)
        new_tables = (all_data[0]['newTables'])
        for table in new_tables:
            if table['create_table']:
                print("Create Table:", table['create_table'])
                _, table_json = getout_table_cells_information([[table['coordinates']['y'], table['coordinates']['x'], (
                        table['coordinates']['y'] + table['coordinates']['height']), (table['coordinates']['x'] +
                                                                                      table['coordinates'][
                                                                                          'width'])]], image,
                                                               evidence)
                new_created_tables.append(ui_format(table_json, table['label']))
                print("CREATED SHOWING TABLE :\n", new_created_tables[-1])
                print("append done")
            else:
                structure = get_textlines(evidence, image)
                print("Create Table:", table['create_table'])

                for all_rows in table['tableRows']:
                    for each_cell in all_rows['cells']:
                        # print('BEFORE VAL :',each_cell['value'])
                        each_cell['value'] = create_from_given_data(structure, [each_cell['coordinates']['y'],
                                                                                each_cell['coordinates']['x'], (
                                                                                        each_cell['coordinates'][
                                                                                            'y'] +
                                                                                        each_cell['coordinates'][
                                                                                            'height']), (
                                                                                        each_cell['coordinates'][
                                                                                            'x'] +
                                                                                        each_cell['coordinates'][
                                                                                            'width'])])
                        # print('AFTER VAL :',each_cell['value'])
                print(table)
                new_created_tables.append(table)

        # all_data[0]['newTables'] = new_created_tables
        save_tables = []
        for each_table in new_created_tables:
            # print(':::',type(all_data[0]['tables']))
            # print('LENGTH : :',len(all_data[0]['tables']))
            each_table['id'] = get_filename() + str(len(all_data[0]['tables']))
            # print("id changed",each_table['id'])
            save_tables.append(
                each_table)  # print('FL:',len(all_data[0]['tables']))  # with open("UI_table_added_evidences.json", "w") as out_file:  #     json.dump(tables_data, out_file)\
        # all_data = list(db.fields.find({"documentId": documentId, "pageId": pageId}))
        all_data[0]['newTables'] = save_tables
        res = client[client_name]['fields'].update_one({"documentId": documentId, "pageId": pageId},
                                                       {'$set': all_data[0]}, upsert=True)
        print('Tables updated')

    except Exception as e:
        print("Exception found on ", e)
        print('No modifications in Tables')
        # print(e)

import lxml.etree as etree
import json
import sys
from os import listdir
from xml.dom.minidom import parseString
import ast
import binascii
from pymongo import MongoClient

import dicttoxml

import cv2
import numpy as np
import json
import math
from copy import deepcopy
from ocr_pattern_hypothesis.utils import frame_utils
from ocr_pattern_hypothesis.frames.basic_frames import Word
from ocr_pattern_hypothesis.frames.structure.engine import StructureEngine
from ocr_pattern_hypothesis.frames.structure.text import TextLine
import time
import os

mongo_ip = "mongodb://localhost"
client_name = "tangoe1"

"""
Created by @amandubey on 22/01/19
"""


def is_coordinates_overlapping(rect1, rect2):
    """
    created by @amandubey on 13/11/18
    returns if two sets of coordinates are overlapping or not
    :param rect1: first set of coordinates
    :param rect2: second set of coordinates
    :return: True when two sets of coordinates are overlapping otherwise false
    """
    left = rect2[2] < rect1[0]
    right = rect1[2] < rect2[0]
    bottom = rect2[3] < rect1[1]
    top = rect1[3] < rect2[1]
    if top or left or bottom or right:
        return False
    else:  # rectangles intersect
        return True


def merge_coordinates(rect1, rect2):
    """
    created by @amandubey on 13/11/18
    returns merged coordinates in format of [T,L,B,R] after merging two coordinates
    :param rect1: first set of coordinates
    :param rect2: second set of coordinates
    :return: merged coordinates in format of [T,L,B,R]
    """
    t = min((rect1[0], rect2[0]))
    l = min((rect1[1], rect2[1]))
    b = max((rect1[2], rect2[2]))
    r = max((rect1[3], rect2[3]))
    return [t, l, b, r]


def check_combined_coordinates(coor_list):
    """
    created by @amandubey on 13/11/18
    combines all those set of coordinates that are overlapping and returns updated list
    :param coor_list: list of all those coordinates that belong to same page
    :return: updated_coor_list
    """
    updated_coor_list = []
    flag = False
    In_flag = False

    for e, item in enumerate(coor_list):
        reccur_list = list.copy(updated_coor_list)
        if len(updated_coor_list) == 0:
            updated_coor_list.append(item)
        else:
            In_flag = False
            for new_num, new_items in enumerate(reccur_list):
                In_flag = is_coordinates_overlapping(new_items, item)
                if In_flag:
                    flag = True
                    coor = merge_coordinates(new_items, item)
                    break
            if not In_flag:
                updated_coor_list.append(item)
            else:
                updated_coor_list[new_num][0] = coor[0]
                updated_coor_list[new_num][1] = coor[1]
                updated_coor_list[new_num][2] = coor[2]
                updated_coor_list[new_num][3] = coor[3]

    if flag:
        return check_combined_coordinates(updated_coor_list)
    else:
        return updated_coor_list


def inv_thresh(original_image, thresh=220):
    """
    Creates inverse image
    """
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return img


def refine_cc_stats(stats, img_area, lowest_area_threshold=10):
    op_stats = []
    for each in stats:
        if not (((each[2] * each[3]) / img_area) > .98):
            op_stats.append(each.tolist())
    return op_stats


def get_connected_components(img: np.ndarray):
    """
    runs Connected Components on inverted images
    :param img: inverted image to be used for finding connected components
    :returns: ndarray of same size as image and array of all the components in format [left,top,width,height,area]
    """
    connectivity = 8
    output = cv2.connectedComponentsWithStats(img, connectivity=connectivity, stats=cv2.CV_16U)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    op_stats = refine_cc_stats(stats, (img.shape[0] * img.shape[1]))
    centroids = output[3]
    labels[labels != 0] = 255

    return labels, op_stats


def get_block_image(image, stats, area_plot=False, page_start=0, page_end=10000):
    """
    Creates block image where area_plot tells either to fill whole height or only block coordinates

    :param image: original image
    :param stats: stats out of  get_connected_components() function to mark all patches present
    :param area_plot: if yes fill the whole row if any white pixel is there in what row
    :param page_start: left coordinate of page that tells the start point for area_plot
    :param page_end: right coordinate of page that tells the last point for area_plot
    :return: image showing pathes as white blocks in black canvas and coordinate list having coordinates of all patches
    """
    block_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    coor_list = []
    for pos in range(1, len(stats)):

        if stats[pos][0] >= page_start and (stats[pos][0] + stats[pos][2]) <= page_end:

            coor_list.append([stats[pos][1], stats[pos][0], (stats[pos][1] + stats[pos][3]),
                              (stats[pos][0] + stats[pos][2])])
            if area_plot:
                block_image[stats[pos][1]:(stats[pos][1] + stats[pos][3]),
                page_start:page_end] = 100

            block_image[stats[pos][1]:(stats[pos][1] + stats[pos][3]),
            stats[pos][0]:(stats[pos][0] + stats[pos][2])] = 255

    # cv2.imshow("SDFG", block_image)
    # if not area_plot:
    #     cv2.imwrite("/home/amandubey/Documents/All Output Images/Paragraphs/"+global_name+"___.jpg", block_image)
    # else:
    #     cv2.imwrite("/home/amandubey/Documents/All Output Images/Paragraphs/"+global_name+"_____.jpg", block_image)
    # # cv2.waitKey(0)
    # print(coor_list)
    return block_image, coor_list


def get_histogram(data: np.ndarray, orientation):
    """
    gives number of black pixels in given orientation

    :param data: ndarray out of which histogram will be calculated
    :param orientation: 1 for each column and 0 for each row
    :return: ndarray of count of black pixels for chosen orientation
    """
    return data.shape[orientation] - np.count_nonzero(data, orientation)


def get_division_point(hist_list, page_height, width, mid_point_thresh=.8):
    """
    In range of 6% of pixels with page mid point in center gives best place for division to two
    :param hist_list: ndarray of count of black pixels
    :param page_height: height of image
    :param mid_pt_thresh: threshold value above which page will be considered as 2 column
    :param width: width of image
    :return: True and index of division point for 2 columns otherwise returns False
    """
    hist_list = hist_list.tolist()
    mid_pt = int(len(hist_list) / 2)
    max_val = max([hist_list[pix] for pix in range((mid_pt - int(.03 * width)), (mid_pt + int(.03 * width)))])
    mid_pt_index = hist_list.index(max_val, (mid_pt - int(.03 * width)), (mid_pt + int(.03 * width)))
    if (max_val / page_height) > mid_point_thresh:
        return True, mid_pt_index
    else:
        return False, mid_pt_index


def get_row_blocks(hist_list, page_start, page_end):
    """
    Using histogram data creates rows with page_start and page_end as left and right coordinates

    :param hist_list: ndarray of count of black pixels
    :param page_start: starting point to start checking rows
    :param page_end: end point to finish checking rows
    :return: list of all rows formed in that segment
    """
    para_block_list = []
    data_start_flag = False
    data_end_flag = True
    for enum, val in enumerate(hist_list):
        if val != max(hist_list) and not data_start_flag:
            data_start_flag = True
            data_end_flag = False
            coor = [enum, page_start, 0, page_end]
        if val == max(hist_list) and data_start_flag:
            data_start_flag = False
            data_end_flag = True
            coor[2] = enum

            para_block_list.append(coor)
    if not data_end_flag:
        coor[2] = enum
        para_block_list.append(coor)
    return para_block_list


def is_inside(rect1, rect2):  # (x1, y1, x1b, y1b), (x2, y2, x2b, y2b)):
    """
    Checks if two rectangles overlap or not
    :return: True if rectangles overlap otheerwise false
    """
    # if check_data():
    left = rect2[3] < rect1[1]  # always -ve
    right = rect2[1] > rect1[3]  # always +ve
    bottom = rect2[0] > rect1[2]  # always +ve
    top = rect2[2] < rect1[0]  # always -ve
    if top or left or right or bottom:
        return False
    else:  # rectangles intersect
        return True


def get_end_paragraph(paragraphs_text, para_block_list):
    """
    Given all paraghaphs and all rows will optimise the results
    :param paragraphs_text: list of inside coordinates
    :param para_block_list: list of blocks formed row wise in get_row_blocks()
    :return: refined list of all patches
    """
    final_para_coor = []
    for block in para_block_list:
        used_data = []
        this_para_coor = []
        for enum, para_coor in enumerate(paragraphs_text):
            if is_inside(block, para_coor):
                used_data.append(enum)
                if len(this_para_coor) < 1:
                    this_para_coor = para_coor
                else:
                    this_para_coor = textline_intersection(this_para_coor, para_coor)

        used_data.sort(reverse=True)
        if len(used_data) > 0:
            final_para_coor.append(this_para_coor)
        for data in used_data:
            paragraphs_text.pop(data)
    return final_para_coor


def refine_pathches(parent_coor, child_coor):
    """
    refines parent_coor by finding if there is atleast one child_coor in each of them
    :param parent_coor: outer coordinates
    :param child_coor: inside coordinates
    :return: refined list from parent_coor that have atleast one child_coor inside
    """
    refined_parent_list = []
    for parent in parent_coor:
        for child in child_coor:
            if is_inside(parent, child):
                refined_parent_list.append(parent)
                break
    return refined_parent_list


def get_pixelwise_histogram(image, orientation):
    """
    refines histogram in a way that data of only those rows/columns remain that has some text in it
    :param image: ndarray out of which histogram will be calculated
    :param orientation: 1 for each column and 0 for each row
    :return: modified histogram list for that orientation
    """

    other_hist_list = get_histogram(image, abs(orientation - 1))
    required_hist_list = get_histogram(image, orientation)
    count = 0
    for val in other_hist_list:
        if val <= int(.02 * image.shape[orientation]):
            count = count + 1
    required_hist_list = required_hist_list - count
    return required_hist_list


def column_division(paragraphs_text, img2, self_divide_columns, num_of_columns=2):
    """
    Gives logic for 1 column page and 2 column page
    """
    # asdf_image=deepcopy(img2)
    image = deepcopy(img2)

    for para in paragraphs_text:
        cv2.rectangle(img2, (para[1], para[0]), (para[3], para[2]), (0, 0, 0), -1)
    final_para_set = []

    inv_img = inv_thresh(img2)
    new = inv_img
    new_inv_img = wrap_image(new, 30, 0)
    # debug_method([],new_inv_img,"inv_img_")
    labels, stats = get_connected_components(new_inv_img)
    block_img, connected_comp_coord = get_block_image(new_inv_img, stats)
    connected_comp_coord = refine_pathches(connected_comp_coord, paragraphs_text)

    hist_list = get_pixelwise_histogram(block_img, 0)
    is_two_column, mid_pt = get_division_point(hist_list, img2.shape[0], img2.shape[1])
    # debug_method([[0,mid_pt,2000,(mid_pt+1)]],deepcopy(image),"MID_POINT")

    if (not is_two_column and self_divide_columns) or num_of_columns == 1:
        page_end = block_img.shape[1]
        page_start = 0
        row_finding_img, _ = (get_block_image(image, stats, True, page_start, page_end))
        hist_list = get_histogram(row_finding_img, 1)
        para_block_list = get_row_blocks(hist_list, page_start, page_end)
        final_para_set.extend(
            get_end_paragraph(connected_comp_coord, para_block_list))

    elif (is_two_column and self_divide_columns) or num_of_columns == 2:
        all_rows_list = []
        for coor in paragraphs_text:
            if mid_pt in range(coor[1], coor[3]):
                all_rows_list.append(coor)

        for ittr in range(0, 2):
            if ittr == 0:
                page_start, page_end = 0, mid_pt - 1
            else:
                page_start, page_end = mid_pt + 1, block_img.shape[1]
            send_image = deepcopy(image)
            send_image[:, 0:page_start] = 255
            send_image[:, page_end:] = 255
            row_finding_img, _ = (get_block_image(send_image, stats, True, page_start, page_end))
            # debug_method([],row_finding_img,"ROW_FIND_IMAGE")
            hist_list = get_histogram(row_finding_img[:, page_start:page_end], 1)
            para_block_list = get_row_blocks(hist_list, page_start, page_end)
            all_rows_list.extend(para_block_list)

        final_para_set.extend(get_end_paragraph(connected_comp_coord, all_rows_list))

    return final_para_set


import re

"""
Gives all textlines
"""


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

        xx = re.findall(r'[a-zA-Z0-9-+=-_]+', label_word)

        if len(xx) < 1:
            label_word = " "

        word_patches_dict[coordinates] = label_word

    try:
        structures = s_engine.run(image, word_args=(word_patches_dict,))
    except IndexError:
        structures = []
    structures = structures.filter(TextLine)
    return structures


def textline_intersection(rect1, rect2):
    """
    Get new coordinates if two blocks are overlapping
    :return: merged coordinates
    """
    t = min((rect1[0], rect2[0]))
    l = min((rect1[1], rect2[1]))
    b = max((rect1[2], rect2[2]))
    r = max((rect1[3], rect2[3]))
    return [t, l, b, r]


"""
logical steps for creating paragraphs
"""


def is_same_paragraph(coor1, coor2, same_para_threshold):
    if coor1[1] in range(coor2[1], coor2[3]) or coor1[3] in range(coor2[1], coor2[3]) or coor2[1] in range(coor1[1],
                                                                                                           coor1[3]) or \
            coor2[3] in range(coor1[1], coor1[3]):
        height_needed = min(abs(coor1[2] - coor1[0]), abs(coor2[2] - coor2[0]))
        height_got = min(abs(coor2[2] - coor1[0]), abs(coor1[2] - coor2[0]))
        if height_got <= int(height_needed * same_para_threshold):
            return True

    return False


def debug_method(para, img, name='_'):
    if len(para) > 0:
        for i in para:
            cv2.rectangle(img, (i[1], i[0]), (i[3], i[2]), (0, 255, 0), 3)

    cv2.imwrite(
        "/home/amandubey/Documents/All Output Images/Paragraphs/" + gn.get_gname() + '_' + name + get_filename() + '.jpg',
        img)
    # cv2.imshow("READ",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def generate_json(final_coord, structure):
    generated_id = 0
    data = []
    all_data = []

    for each_coord in final_coord:
        c = [0, 0, 0, 0]
        unsorted_text_data = []
        text_data = ''
        for enum, textline in enumerate(structure):
            c[0] = (textline.coordinates[0][1])
            c[1] = (textline.coordinates[0][0])
            c[2] = (textline.coordinates[1][1])
            c[3] = (textline.coordinates[1][0])
            if is_inside(each_coord, c):
                newString = (str(textline).encode('ascii', 'ignore')).decode("utf-8")
                unsorted_text_data.append([c[0], (newString + "\\n")])
        unsorted_text_data.sort()
        # print(unsorted_text_data)
        #
        for each_str in unsorted_text_data:
            text_data = text_data + str(each_str[1])
        all_data.append({'x': each_coord[1], 'y': each_coord[0], 'width': (each_coord[3] - each_coord[1]),
                         'height': (each_coord[2] - each_coord[0]), 'value': text_data})
    all_sorted_data = sorted(all_data, key=lambda x: x['y'])
    for each_sorted_data in all_sorted_data:
        generated_id = generated_id + 1
        data.append({"id": generated_id, "coord": {'x': each_sorted_data['x'], 'y': each_sorted_data['y'],
                                                   'width': each_sorted_data['width'],
                                                   'height': each_sorted_data['height']},
                     'value': each_sorted_data['value']})

    return data


"""
Function to call paragraph detection
"""


def detect_paragraph(org_image, evidence, same_para_threshpld=1.5, self_divide_columns=True, num_of_columns=0):
    noiseless_image = remove_noise(org_image)
    img = wrap_image(noiseless_image, 30, 255)
    structure = get_textlines(evidence, img)
    c = [0, 0, 0, 0]
    textline_coor = []
    paragraphs_text = []
    used_enum_list = []
    img2 = deepcopy(img)
    for enum, textline in enumerate(structure):
        used_enum_list.append(enum)
        c[0] = (textline.coordinates[0][1])
        c[1] = (textline.coordinates[0][0])
        c[2] = (textline.coordinates[1][1])
        c[3] = (textline.coordinates[1][0])

        cv2.rectangle(img, (c[1], c[0]), (c[3], c[2]), (0, 0, 0), 3)  #####  Show Text line
        textline_coor.append([c[0], c[1], c[2], c[3]])
    # cv2.imwrite("/home/amandubey/Documents/textlines.jpg",img)
    # exit()
    # debug_method([],img,"textlines")
    for text_coor in (textline_coor):
        if len(paragraphs_text) == 0:
            paragraphs_text.append(text_coor)
        else:
            # debug_method(paragraphs_text,deepcopy(img))
            inside_paragraph = False
            for enum, para_coor in enumerate(paragraphs_text):
                if is_same_paragraph(text_coor, para_coor, same_para_threshpld):
                    paragraphs_text[enum] = textline_intersection(text_coor, para_coor)
                    inside_paragraph = True
                if inside_paragraph:
                    break
            if not inside_paragraph:
                paragraphs_text.append(text_coor)
    final_paragraphs_text = check_combined_coordinates(paragraphs_text)
    # debug_method(final_paragraphs_text,deepcopy(img2),"after_comb_coor")

    final_paragraphs_text = column_division(final_paragraphs_text, deepcopy(img2), self_divide_columns, num_of_columns)
    # debug_method(final_paragraphs_text,deepcopy(img2),"after_col")

    final_paragraphs_text = check_combined_coordinates(final_paragraphs_text)
    json_data = generate_json(final_paragraphs_text, structure)

    return json_data


def get_tight_bounds(image, top, left):
    inv_img = inv_thresh(image)
    row_hist = get_histogram(inv_img, 1)
    t = top + (np.where(row_hist != image.shape[1])[0][0])
    b = top + (np.where(row_hist != image.shape[1])[0][-1])
    col_hist = get_histogram(inv_img, 0)
    l = left + (np.where(col_hist != image.shape[0])[0][0])
    r = left + (np.where(col_hist != image.shape[0])[0][-1])
    # cv2.imshow("bound",inv_img)

    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return [t, l, b, r]


def wrap_image(image, threshold=10, insert_value=0):
    wrap_image = image.copy()

    wrap_image[0:threshold, :] = insert_value
    wrap_image[:, 0:threshold] = insert_value
    wrap_image[(wrap_image.shape[0] - threshold):, :] = insert_value
    wrap_image[:, (wrap_image.shape[1] - threshold):] = insert_value
    return wrap_image


def remove_noise(img):
    """
    Takes a numpy array representing an image as input and removes possible
    :param img: A 3-dimensional numpy array representing an image
    :return:
    """

    # smooth the image with alternative closing and opening
    # with an enlarging kernel
    morph = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    morph = cv2.erode(morph, kernel1, iterations=1)

    # split the gradient image into channels
    image_channels = np.split(np.asarray(morph), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(image_channels[i], 127, 255, cv2.THRESH_OTSU)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

    # merge the channels
    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)

    # save the denoised image
    return image_channels


######################################################################################################################

class k():
    def __init__(self):
        self.key = 0

    def get_key(self):
        self.key = self.key + 1
        return self.key


delta = k()

all_fields = {}
relationship_pairs = {}


def collect_children(page_id, parents):
    child_list = []
    for each_child in parents:
        if each_child['field']:
            save_name = str(page_id + '_' + each_child['_id'])

        else:
            save_name = str(each_child['_id'])
        child_list.append(save_name)
        relationship_pairs[str(delta.get_key())] = {'parent': save_name,
                                                    'child': collect_children(page_id, each_child['children']),
                                                    'is_link': not (each_child['field'])}

    return child_list


def clean_relationship_pairs(relationship_pairs):
    parents_list = []
    keys_poped = []
    links_list = []
    for k, v in relationship_pairs.items():

        if v['parent'] not in parents_list:
            parents_list.append(v['parent'])
            if v['is_link']:
                links_list.append(str(1 + parents_list.index(v['parent'])))
        else:
            indx = parents_list.index(v['parent'])
            relationship_pairs[str(1 + indx)]['child'].extend(v['child'])
            keys_poped.append(k)
    for k in keys_poped:
        relationship_pairs.pop(k)
    relationship_pairs = clean_links(relationship_pairs, links_list)

    return relationship_pairs


def clean_links(relationship_pairs, links_list):
    for enum in links_list:
        for each_enum, val in relationship_pairs.items():
            if relationship_pairs[enum]['parent'] in val['child']:
                val['child'].extend(relationship_pairs[enum]['child'])
                val['child'].pop(val['child'].index(relationship_pairs[enum]['parent']))
    for k in links_list:
        relationship_pairs.pop(k)
    return relationship_pairs


def collect_all_fileds(page_id, fields, id_value):
    for each_data in fields:
        # print(each_data)
        each_data['children'] = []
        id_value = id_value + 1
        key_val = ast.literal_eval(each_data['value'])
        dict_xml = {'id': str(id_value), 'tag': each_data['tag'], 'type': each_data['type'], 'key': key_val[0]['key'],
                    'value': key_val[1]['key'], 'children': []}
        all_fields[page_id + '_' + each_data['_id']] = dict_xml
    return id_value


def create_relation_fields(all_fields, relationship_pairs):
    poped_ids = []
    field_list = []
    all_fields = fetch_all_parents(all_fields, relationship_pairs)
    for key_enum, tree_relation in relationship_pairs.items():
        for child_id in tree_relation['child']:
            all_fields[tree_relation['parent']]['children'].append(all_fields[child_id])
            poped_ids.append(child_id)
    for popid in poped_ids:
        all_fields.pop(popid)
    print("RELATED FIELDS SHOWING HERE :", all_fields)
    for k, v in all_fields.items():
        field_list.append(v)
    print("ALL RELATED FIELDS SHOWING HERE :", field_list)
    return field_list


def get_all_fields(mongo_ip, client_name, document_id):
    client = MongoClient(mongo_ip)
    db = client[client_name]
    relations = list(db.pageRelations.find({"document_id": document_id}))
    fields = list(db.fields.find({"documentId": document_id}))
    # print("DOCUMENT DETAILS:::::::", document_id)
    # print("ALL REALTIONS	:	", relations)
    for enum, val in enumerate(relations):
        collect_children(val["page_id"], val["relations"])
    id_value = 0
    clean_relationship_pairs(relationship_pairs)
    # print("Cleansed rellationship :", relationship_pairs)
    # print("ALL CAPTURED FIELDS ARE ::", fields)
    for enum, val in enumerate(fields):
        id_value = collect_all_fileds(val["pageId"], val['fields'], id_value)
    field_list = create_relation_fields(all_fields, relationship_pairs)
    return field_list


#######################################################################################################################

def remove_unicodes_here(val):
    pattern = re.compile(
        r"(?![A-Za-z]|\d|\s|\.|\:|\%|\\|\)|\(|\"|\@|\!|\#|\$|\%|\^|\*|\-|\+|\_|\=|\{|\}|\[|\]|\;|\'|\?|\>|\<|\,|\`|\~|\||\/).")
    xx = pattern.split(val)
    if len(xx) > 1:
        val = ''.join(xx)
    return val.strip()

def get_amount_value(amnt_str):
    amnt_str =amnt_str.strip()
    if re.search("^c?r?[\-\$\s]?[\-\$\s]?[\-\$\s]?[\-\$\s]?[\d\.]{1,}[\d,\.]{1,}c?r?$",amnt_str,flags=re.IGNORECASE):
        # print("true")
        amt = (re.search("^c?r?[\-\$\s]?[\-\$\s]?[\-\$\s]?[\-\$\s]?([\d\.]{1,}[\d,\.]{1,})c?r?$", amnt_str,
                         flags=re.IGNORECASE).groups()[0])
        if re.search("(cr|.*\-.*[\-\$\s]?[\-\$\s]?[\-\$\s]?[\-\$\s]?[\d\.]{1,}[\d,\.]{1,})", amnt_str,
                     flags=re.IGNORECASE):
            negative_amt = '-' + amt
            # print("negative",type(negative_amt),negative_amt)
            return float(negative_amt)
        else:
            # print("positive",type(amt),amt)
            return float(amt)
    else:
        # print('false')
        return 0

def clean_tables(tables_data):
    all_tables = []
    sub_amount_calculated_list=[]
    merged_id = 0
    total_amount_calculated=0
    for enum, each_page in enumerate(tables_data):
        for each_table in each_page:
            sub_total_amount_calculated=0
            table = {"id": merged_id, 'tag': each_table['tag'],'label': each_table['label'], 'rows': []}
            for each_row in each_table['tableRows']:
                # print("XML ROW DATA", each_row)
                row_data = {}
                last_column = ''
                for each_cell in each_row['cells']:
                    if len(row_data.keys()) == 0:
                        row_data['description'] = each_cell['value']
                    row_data['amount'] = each_cell['value']
                table['rows'].append(row_data)
                sub_total_amount_calculated+=get_amount_value(row_data['amount'])
            sub_amount_calculated_list.append(sub_total_amount_calculated)

            merged_id = merged_id + 1
            all_tables.append(table)
            total_amount_calculated+=sub_total_amount_calculated

    return all_tables,sub_amount_calculated_list,total_amount_calculated


def get_table_data_xml(mongo_ip, client_name, document_id, tag_id):
    client = MongoClient(mongo_ip)
    db = client[client_name]
    # relations = []
    tables_data = []
    fields = list(db.fields.find({"documentId": document_id}))
    for each_field in fields:
        tables_data.append(each_field['tables'])
    all_tables,sub_amount_calculated_list,total_amount_calculated = clean_tables(tables_data)
    tables_xml = ''
    for enum,info in enumerate(all_tables):
        tables_xml += '<line tag_id="' + str(tag_id) + '" item="' + str(info['label']) +'" sub_charges="'+str(sub_amount_calculated_list[enum]) +'">'
        tag_id = tag_id + 1
        for col in info['rows']:
            # print("COLUMN IN TABLES", col)
            tables_xml += '<charges tag_id="' + str(tag_id) + '" description=" ' + remove_unicodes_here(
                col['description']) + '" amount=" ' + remove_unicodes_here(col['amount'])
            # for col_key, col_val in col.items():
            #     tables_xml += ' ' + col_key + ' = "' + col_val + '"'
            tables_xml += '"/>'
            tag_id = tag_id + 1
        tables_xml += '</line>'
    return tables_xml, tag_id,total_amount_calculated


def fetch_all_parents(all_fields, relationship_pairs):
    parents_list = []
    rel_based_id = 1
    for k, each_rel in relationship_pairs.items():
        parents_list.append(each_rel['parent'])
    poping_list = []
    for key, field in all_fields.items():
        if key not in parents_list:
            poping_list.append(key)
    for key in poping_list:
        all_fields.pop(key)
    return all_fields


def get_child(all_children):
    all_third_lvl_children = []
    for each_child in all_children:
        all_third_lvl_children.append(each_child)

        if len(each_child['children']) > 0:
            all_third_lvl_children.extend(get_child(each_child['children']))
            each_child['children'] = []
    return all_third_lvl_children


def create_3_lvl_relation(all_fields):
    poped_enum = []
    for enum, each_field in enumerate(all_fields):
        print(type(each_field['children']), each_field)
        for each_child in each_field['children']:
            # if each_child['children']>0:
            if len(each_child['children']) > 0:
                each_child['children'] = get_child(each_child['children'])
            # for second_lvl_child in each_child['children']:
            #    print("SECOND LVL CHILD",second_lvl_child)
            #    if len(second_lvl_child) > 0:
            #        print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
            #        second_lvl_child['children'] = get_child(second_lvl_child['children'])
    # poped_enum.sort(reverse=True)
    # for i in poped_enum:
    #   all_fields.pop(i)
    return all_fields


def get_all_relation_fields(all_children):
    all_relation_fields = []
    # print(all_children)

    for each_child in all_children:
        all_relation_fields.append([each_child['tag'], each_child['value']])

        if len(each_child['children']) > 0:
            all_relation_fields.extend(get_all_relation_fields(each_child['children']))
            each_child['children'] = []
    return all_relation_fields


def insert_value_in_field_dict(secondary_dict, primary_dict, each_field):
    if each_field[0] in primary_dict.keys():
        if len(primary_dict[each_field[0]]) == 0:
            primary_dict[each_field[0]] = each_field[1]
        elif remove_unicodes_here(primary_dict[each_field[0]]) != remove_unicodes_here(each_field[1]):
            # print("ZZZZZZZZZZZZZ",primary_dict[each_field[0]],each_field[1])
            if each_field[0] not in secondary_dict.keys():
                secondary_dict[each_field[0]] = []
            if each_field[1] not in secondary_dict[each_field[0]]:
                secondary_dict[each_field[0]].append(each_field[1])
    return primary_dict, secondary_dict


def fetch_all_dict(invoice_info_dict, invoice_header_dict, lvl3_res):
    remaining_tags = {}
    secondary_dict = {}
    list_of_fields = []
    for field in lvl3_res:
        list_of_fields.extend(get_all_relation_fields(lvl3_res))
    # print("list_of_fields",list_of_fields)
    for each_field in list_of_fields:
        if each_field[0] in invoice_header_dict.keys():
            invoice_header_dict, secondary_dict = insert_value_in_field_dict(secondary_dict, invoice_header_dict,
                                                                             each_field)

        elif each_field[0] in invoice_info_dict.keys():
            invoice_info_dict, secondary_dict = insert_value_in_field_dict(secondary_dict, invoice_info_dict,
                                                                           each_field)

        else:
            if each_field[0] not in remaining_tags.keys():
                remaining_tags[each_field[0]] = each_field[1]
            else:
                remaining_tags, secondary_dict = insert_value_in_field_dict(secondary_dict, remaining_tags, each_field)

    return invoice_info_dict, invoice_header_dict, secondary_dict, remaining_tags


def create_xml_for_fields(mongo_ip, client_name, document_id, invoice_info_dict, invoice_header_dict, secondary_dict,
                          remaining_tags):
    tag_id = 1
    xml_output = "<invoice"
    for tag, value in invoice_header_dict.items():
        xml_output += ' ' + str(tag) + '=' + '"' + remove_unicodes_here(str(value)) + '"'
    xml_output += ' tag_id="' + str(tag_id) + '">'
    tag_id = tag_id + 1
    xml_output += '<invoice_info tag_id="' + str(tag_id) + '">'
    for tag, value in invoice_info_dict.items():
        xml_output += '<charge tag_id="' + str(tag_id) + '" amount="' + remove_unicodes_here(
            value) + '" type="' + tag + '"/>'
        tag_id = tag_id + 1
    xml_output += '</invoice_info> <invoice_details tag_id="' + str(tag_id) + '">'

    table_xml, tag_id, total_amount_calculated = get_table_data_xml(mongo_ip, client_name, document_id, tag_id + 1)
    xml_output += table_xml
    xml_output += '<line tag_id="' + str(tag_id) + '" item="remaining_tags">'
    tag_id = tag_id + 1
    for tag, value in remaining_tags.items():
        xml_output += '<remaining tag_id="' + str(tag_id) + '" ' + tag + '="' + remove_unicodes_here(value) + '"/>'
        tag_id = tag_id + 1
    xml_output += '</line><line tag_id="' + str(tag_id) + '" item="duplicate_tags">'
    tag_id = tag_id + 1
    for tag, value in secondary_dict.items():
        xml_output += '<duplicates tag_id="' + str(tag_id) + '" ' + tag + '="' + remove_unicodes_here(
            ' / '.join(value)) + '"/>'
        tag_id = tag_id + 1
    xml_output += '</line>'
    xml_output += '</invoice_details><validation_result='
    if total_amount_calculated==invoice_info_dict['total_current_charges']:
        xml_output+='"true"/>'
    else:
        xml_output+='"false" value="'+str(total_amount_calculated)+'"/>'
    xml_output+='</invoice>'

    return xml_output


def combine_json_parse_xml(uploadpath):
    data_list = []
    data = {}
    document_id = uploadpath.split('/')[-2]
    print("UPLOAD PATH IS ::::", uploadpath)
    path = uploadpath + "/fields/"
    for f in listdir(path):
        with open(uploadpath + "words/" + f) as ev:
            evidence = ast.literal_eval(ev.read())
        img_name = re.sub('.json$', '.jpg', f)
        # print(uploadpath+'/images/'+img_name)
        image = cv2.imread(uploadpath + 'images/' + img_name)
        with open(path + f) as fs:
            temp = (json.loads(fs.read()))
            # temp["textIslands"]=detect_paragraph(image, evidence, 1.5,False,2)
            temp.pop('fields')
            data_list.append(temp)
    data['all_Fields'] = get_all_fields(mongo_ip, client_name, document_id)
    data['page_Data'] = data_list

    # table_xml = get_table_data_xml(mongo_ip, client_name, document_id)

    lvl3_res = create_3_lvl_relation(data['all_Fields'])
    print("lvl3_res", lvl3_res)

    invoice_header_dict = {"country": '', "zip": '', "state": '', "city": '', "address": '', "number": '',
                           "due_date": '', "date": '', "amount_due": '', "account": '', "currency": '',
                           "vendor_id": '', "invoice_type": ''}

    invoice_info_dict = {'total_current_charges': '', 'late_charges': '', 'PDB': ''}
    invoice_info_dict, invoice_header_dict, secondary_dict, remaining_tags = fetch_all_dict(invoice_info_dict,
                                                                                            invoice_header_dict,
                                                                                            lvl3_res)
    xml_output = create_xml_for_fields(mongo_ip, client_name, document_id, invoice_info_dict, invoice_header_dict,
                                       secondary_dict, remaining_tags)
    print("XML", xml_output)
    with open(str(uploadpath) + "/" + "data.xml", "w") as fs:
        fs.write(xml_output)


if __name__ == '__main__':
    uploadpath = sys.argv[1]
    # uploadpath = "/home/amandubey/Downloads/5c530a120ed0a632bcaf797c/"
    combine_json_parse_xml(uploadpath)
    print("success")

# paragraphs_text = detect_paragraph(img, evidence, 1.5,False,2)

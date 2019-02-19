import json
import cv2
import os
import ast
import sys
from copy import deepcopy
from ocr_pattern_hypothesis.utils import frame_utils
from ocr_pattern_hypothesis.frames.basic_frames import Word
from copy import deepcopy

# STRUCTURE FRAMES
from ocr_pattern_hypothesis.frames.structure.engine import StructureEngine
from ocr_pattern_hypothesis.frames.structure.text import Paragraph, TextLine

# CONTENT FRAMES
from ocr_pattern_hypothesis.frames.content.engine import ContentEngine
from ocr_pattern_hypothesis.frames.content.where_rules import where_page, where_position, close_by
from ocr_pattern_hypothesis.frames.content.what_rules import MatchSRL, LookUp, HasKey, Near
from ocr_pattern_hypothesis.frames.content.what_group_rules import GrMatchSRL, GrLookUp, GrHasKey, \
    GrNear

s_engine = StructureEngine((
    TextLine.generate,
    # Paragraph.generate,
    # Table.generate
))
what_group_rules = {
    'match-srl': GrMatchSRL,
    "lookup": GrLookUp,
    "has-key": GrHasKey,
    "near": GrNear
}

what_rules = {
    'match-srl': MatchSRL,
    "lookup": LookUp,
    "has-key": HasKey,
    "near": Near

}
where_rules = {
    'page': where_page,
    'position': where_position,
    "close-by": close_by
}


def format_data(all_results, tb, rules, pageno):
    # print(tb)
    type_dict = {}
    for norm_tag, r in rules.items():
        if "has-key" in r["hints"]["what"].keys():
            type_dict[norm_tag] = "Key-value pair"
        else:
            type_dict[norm_tag] = "Standalone"

    final_formated_data = {}
    final_formated_data["pageNumber"] = pageno - 1
    final_formated_data["fields"] = []
    final_formated_data["confidenceScore"] = 0

    for index, frame in enumerate(all_results[0]):
        temp_d = {}
        temp_d["id"] = index
        temp_d["tag"] = frame.name
        temp_d["confidenceScore"] = frame.confidence
        temp_d["coord"] = {
            "x": frame.coordinates[0][0],
            "y": frame.coordinates[0][1],
            "width": int(frame.coordinates[1][0]) - int(frame.coordinates[0][0]),
            "height": int(frame.coordinates[1][1]) - int(frame.coordinates[0][1])
        }
        temp_d["type"] = type_dict[frame.name]
        temp_key = "Not-Available"
        if temp_d["type"] == "Key-value pair":

            # print(tb[frame.name][0][tuple(frame.contains['words'])])
            try:

                for g in (tb[frame.name][0][tuple(frame.contains['words'])]):
                    if g[3] == "has-key":
                        print("gggggg", g)
                        temp_key = g[2]["matched_key"]
            except KeyError:
                temp_key = frame.name
                print("key not found")
                # print("g",(tb[frame.name][0][(tuple(frame.contains['words']), frame.coordinates)]))

            temp_d["value"] = str({"key": temp_key, "value": Word.join(*frame.contains['words'])})
        elif temp_d["type"] == "Standalone":
            temp_d["value"] = Word.join(*frame.contains['words'])
        final_formated_data["fields"].append(temp_d)
        final_formated_data["confidenceScore"] = final_formated_data["confidenceScore"] + frame.confidence

    try:
        final_formated_data["confidenceScore"] = final_formated_data["confidenceScore"] / len(all_results[0])
    except ZeroDivisionError:
        final_formated_data["confidenceScore"] = 0

    return final_formated_data


def get_page_hypothesis(image, evidence, rules):
    c_engine = ContentEngine(rules, what_rules, where_rules, what_group_rules)

    word_patches_dict = {}

    for word in evidence["words"]:
        label = word["label"]
        coordinates = (

            word["coordinates"]["y"], word["coordinates"]["x"],
            word["coordinates"]["y"] + word["coordinates"]["height"],
            word["coordinates"]["x"] + word["coordinates"]["width"]
        )
        word_patches_dict[coordinates] = label
    # print()

    structures = s_engine.run(image, word_args=(word_patches_dict,))
    all_results, tb = c_engine.run([image], [structures])
    return (format_data(all_results, tb, rules, evidence["pageNumber"]))



    # for page, img in enumerate(imgs, 1):
    #     try:
    #         structures = s_engine.run(img, word_args=(word_patches_dict,))
    #     except IndexError:
    #         structures = []
    #     all_structures.append(structures)
    #
    # all_results,tb = c_engine.run(imgs, all_structures)
    #
    # # for k,v in tb[0].items():
    # #     print(k)
    # #     print(type(k))
    # #     break
    # # # print(tb)
    # # # exit()
    #
    # print(format_data(all_results,tb,rules,evidence["pageNumber"]))
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


def mark_positions(image):
    position_clause={
    'top-half':[0,0,int(image.shape[1]/2),image.shape[0]],
    'bottom-half':[int(image.shape[1]/2),0,int(image.shape[1]),image.shape[0]],
    'left-half':[0,0,int(image.shape[1]),int(image.shape[0]/2)],
    'right-half':[0,int(image.shape[0]/2),int(image.shape[1]),int(image.shape[0])],
    'top-left-corner':[0,0,int(image.shape[1]/2),int(image.shape[0]/2)],
    'top-right-corner':[0,int(image.shape[0]/2),int(image.shape[1]/2),int(image.shape[0])],
    'bottom-left-corner':[int(image.shape[1]/2),0,int(image.shape[1]),int(image.shape[0]/2)],
    'bottom-right-corner':[int(image.shape[1]/2),int(image.shape[0]/2),int(image.shape[1]),int(image.shape[0])],
    'all':[0,0,image.shape[1],image.shape[0]]
    }
    return position_clause

def filter_by_coordinates(field_list,rules,position_clause):
    poped_enum=[]
    print("ALLRULES",rules)
    for field_enum,each_field in enumerate(field_list):
        inside_coordinates_flag=False
        #print("FIELDS ::",each_field)
        #print("EACH_RULES :",each_field['tag'])
        #print(rules[each_field['tag']]['hints']['where'])
        # print("ALL",position_clause['all'])
        for each_pos in rules[each_field['tag']]['hints']['where']['position']['values']:
            if is_inside(position_clause[each_pos], [each_field['coord']['y'], each_field['coord']['x'],
                                                     (each_field['coord']['y'] + each_field['coord']['height']),
                                                     (each_field['coord']['x'] + each_field['coord']['width'])]):
                inside_coordinates_flag=True
                break
        if not inside_coordinates_flag:
            poped_enum.append(field_enum)
    poped_enum.sort(reverse=True)
    # print('POPED ENUM',poped_enum)
    for enum in poped_enum:
        field_list.pop(enum)
    #print("FIELD_LIST ARE ",field_list)
    return field_list


def filter_on_where_rules(im,page_results,rules):
    position_clause = mark_positions(im)
    page_results['fields']=filter_by_coordinates(page_results['fields'],rules,position_clause)
    return page_results


if __name__ == "__main__":
    input_list = sys.argv
    image_folder=input_list[1] + "images/"
    # image_folder = input_list[1]
    evidence_folder = input_list[1] + "words/"
    output_folder = input_list[1] + "fields/"
    rule_json = input_list[1] + "hints.json"

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # with open(rule_json) as fp:
    #     str_ = fp.readline()
    #print("RULE HERE:",rule_json)
    rules = json.load(open(rule_json))
    #print(rules)
    rules2=deepcopy(rules)

    for k,v in rules.items():
        v['hints']['where']={}
    # rules[]['where']={}
    #print("RULE1 :",rules)
    #print("RULE2 :",rules2)

    # print(rules.keys())
    # exit()

    # doc_level_hypothesis=[]
    page_count=1
    for evidence_name in os.listdir(evidence_folder):
        print("PAGE COUNT :",page_count)
        page_count=page_count+1
        if evidence_name.startswith("."):
            continue

        # print(image_name)
        with open(evidence_folder + evidence_name) as fp:
            str_evidence = fp.readline()

        evidence = ast.literal_eval(str_evidence)
        #print("EVIDENCES :",evidence)
        print("img: " + image_folder + evidence_name[:-4].replace("_page", "") + "jpg")
        im = cv2.imread(image_folder + evidence_name[:-4].replace("_page", "") + "jpg")
        #print("startloading")
        #print(type(im))
        #print("image loaded")

        # page_results = {"pageNumber": 0, "fields": [{"id": 0, "tag": "Account Number", "confidenceScore": 1.0, "coord": {"x": 838, "y": 237, "width": 74, "height": 26}, "type": "Key-value pair", "value": "{'key': 'account no', 'value': '112227'}"}], "confidenceScore": 1.0}
        page_results = get_page_hypothesis(image=im, evidence=evidence, rules=rules)
        print(page_results)
        print('---------------------------------------------------------------------------------')
        page_results = filter_on_where_rules(im, page_results, rules2)
        json.dump(page_results, open(output_folder + (evidence_name), "w"))
        # doc_level_hypothesis.append(page_results)
        # print(json.load(open(evidence_folder+image_name[:-3] + "JSON")))
        # break
        # print("****************************************")
        # print(doc_level_hypothesis)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")



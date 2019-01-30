import numpy as np
from IDP_pipeline.ocr.utils import cvutils
from IDP_pipeline.runner.rzt.Assemble_Evidences import assemble_evidences
from copy import deepcopy
from ocr_pattern_hypothesis.utils import frame_utils
from ocr_pattern_hypothesis.frames.basic_frames import Word

# STRUCTURE FRAMES
from ocr_pattern_hypothesis.frames.structure.engine import StructureEngine
from ocr_pattern_hypothesis.frames.structure.text import Paragraph, TextLine

# CONTENT FRAMES
from ocr_pattern_hypothesis.frames.content.engine import ContentEngine
from ocr_pattern_hypothesis.frames.content.where_rules import where_page, where_position, close_by
from ocr_pattern_hypothesis.frames.content.what_rules import MatchSRL, LookUp, HasKey, Near
from ocr_pattern_hypothesis.frames.content.what_group_rules import GrMatchSRL, GrLookUp, GrHasKey, \
    GrNear

import cv2
import imutils
from IDP_pipeline.ocr.ocr_entities.rzt_ocr.rzt_ocr_entity import RZT_OCR
from IDP_pipeline.ocr.ocr_entities.rzt_ocr.word_detector import RZTWordDetector
import os
import tensorflow as tf
from IDP_pipeline.ocr.ocr_entities.rzt_ocr.lstm_prediction_model import PredictionModel as PredictionModel
from IDP_pipeline.ocr.ocr_entities.google_cloud_vision.gv_ocr_entity import Google_Cloud_Vision_OCR
import json
import glob
import re


def format_data(all_results, tb, rules, pageno):
    type_dict = {}
    for norm_tag, r in rules.items():
        if "has-key" in r["hints"]["what"].keys():
            type_dict[norm_tag] = "Key-value pair"
        else:
            type_dict[norm_tag] = "Standalone"

    final_formated_data = {}
    final_formated_data["pageNumber"] = pageno
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
            try:
                for g in (tb[frame.name][0][(tuple(frame.contains['words']), frame.coordinates)]):
                    if g[3] == "has-key":
                        temp_key = g[2]["matched_key"]
            except KeyError:
                temp_key = frame.name

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


def hypothesis(evidence, image, rules, page_no):
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

    c_engine = ContentEngine(rules, what_rules, where_rules, what_group_rules)

    # Load the image!
    imgs = [image]
    word_patches_dict = {}
    for entry in evidence['words']:
        c = entry["coordinate"]
        label = entry['label']

        coordinates = (
            c['y'], c['x'],
            c['y'] + c['height'], c['x'] + c['width']
        )
        word_patches_dict[coordinates] = label
    all_structures = []
    for page, img in enumerate(imgs, 1):
        try:
            structures = s_engine.run(img, word_args=(word_patches_dict,))
        except IndexError:
            structures = []
        all_structures.append(structures)

    all_results, tb = c_engine.run(imgs, all_structures)
    return format_data(all_results, tb, rules, page_no)


if __name__ == "__main__":

    import sys
    root_folder = sys.argv[1]
    rule_json = sys.argv[2]

    rules = json.load(open(rule_json))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.2
    session = tf.InteractiveSession(config=config_sess)
    prediction_model = PredictionModel(
        model_dir=None,
        session=session)

    evidence_folder = root_folder + "words/"
    text_image_folder = root_folder + "images/"
    fields_json_folder = root_folder + "fields/"

    if not os.path.isdir(fields_json_folder):
        os.mkdir(fields_json_folder)

    for filepath in glob.glob(text_image_folder + "*"):
        im = cv2.imread(filepath)
        evidence_file = filepath.split('/')[-1]
        evidence_file = re.sub('.jpg$', '.json', evidence_file)
        with open(evidence_folder + evidence_file) as f:
            assembled_evidence = json.load(f)
        formatted_fields = hypothesis(evidence=assembled_evidence, image=im, rules=rules,
                                      page_no=int((filepath.split('_')[-1]).replace('.jpg', '')))
        with open(fields_json_folder + evidence_file, "w") as evfile:
            json.dump(formatted_fields, evfile)

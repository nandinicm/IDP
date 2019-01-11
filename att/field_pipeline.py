import numpy as np
from IDP_pipeline.ocr.utils import cvutils
from IDP_pipeline.runner.rzt.Assemble_Evidences import assemble_evidences
from copy import deepcopy
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
    for k, v in evidence['evidence_words'].items():
        c = v["assembled_result"][0]
        label = v["assembled_result"][1]

        coordinates = (
            c[0], c[1],
            c[2], c[3]
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

    root_folder = "/home/rztuser/IDP/run_result/"
    image_folder = "/home/rztuser/IDP/images/"
    rule_json = "/home/rztuser/IDP/Jsons/att_rules_new1.json"
    run_evidence = False

    rules = json.load(open(rule_json))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.2
    session = tf.InteractiveSession(config=config_sess)
    prediction_model = PredictionModel(
        model_dir=None,
        session=session)

    evidence_folder = root_folder + "evidence/"
    fields_json_folder = root_folder + "fields_json/"

    if not os.path.isdir(fields_json_folder):
        os.mkdir(fields_json_folder)

    for filepath in glob.glob(image_folder + "*"):

        with open(filepath, "rb") as binfile:
            pdf_words, document_name = fetch_words(binfile.read(), filepath)

            for page_key, val in pdf_words.items():
                page_file = document_name + "_" + page_key + ".json"
                im = val["numpy_image"]
                if not run_evidence and os.path.isfile(evidence_folder + page_file):
                    with open(evidence_folder + page_file, "r") as evfile:
                        assembled_evidence = json.load(evfile)
                else:
                    text_im = deepcopy(im)
                    evidence_list = []
                    text_patch_list = val["text_images"]
                    if "rzt_ocr" in required_evidences:
                        rzt_evidences = {}
                        for text_patch_key in text_patch_list:
                            t = eval(text_patch_key.split("_")[0])
                            orientation = text_patch_key.split("_")[1]
                            tp = im[t[1]:t[3], t[0]:t[2]].copy()
                            color_ = (0, 0, 255)
                            if orientation == "V":
                                color_ = (255, 0, 0)

                            image_gray = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
                            image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                            if cvutils.is_inverted_text_patch(image_bin):
                                tp = (255 - tp).astype(np.uint8)

                            if orientation == "V":
                                tp = imutils.rotate_bound(tp, 90)

                            rzt_evidence = RZT_OCR(image=tp.copy(), lstm_model=prediction_model,
                                                   word_detector=RZTWordDetector.get_words_using_word_space_cluster).get_word_coordinates_with_string()

                            rzt_evidences[text_patch_key] = rzt_evidence
                        evidence_list.append(rzt_evidences)

                    if "tesseract" in required_evidences:
                        tesseract_evidences = {}
                        for text_patch_key in text_patch_list:
                            t = eval(text_patch_key.split("_")[0])
                            orientation = text_patch_key.split("_")[1]

                            tp = im[t[1]:t[3], t[0]:t[2]].copy()

                            image_gray = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
                            image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                            if cvutils.is_inverted_text_patch(image_bin):
                                tp = (255 - tp).astype(np.uint8)

                            if orientation == "V":
                                tp = imutils.rotate_bound(tp, 90)
                            tesseract_json = tesseract_evidence(tp)

                            tesseract_evidences[text_patch_key] = {
                                "tesseract_words": tesseract_json["tesseract_content"]}
                        evidence_list.append(tesseract_evidences)
                    if "google_cloud_vision" in required_evidences:
                        gv_evidences = {}
                        for text_patch_key in text_patch_list:
                            t = eval(text_patch_key.split("_")[0])
                            orientation = text_patch_key.split("_")[1]

                            tp = im[t[1]:t[3], t[0]:t[2]].copy()

                            image_gray = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
                            image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                            if cvutils.is_inverted_text_patch(image_bin):
                                tp = (255 - tp).astype(np.uint8)

                            if orientation == "V":
                                tp = imutils.rotate_bound(tp, 90)

                            gv_evidence = Google_Cloud_Vision_OCR(image=tp).get_word_coordinate_with_string()

                            gv_evidences[text_patch_key] = gv_evidence
                        evidence_list.append(gv_evidences)

                    assembled_evidence = assemble_evidences(im, evidence_list, val["words"], text_patch_list,
                                                            required_evidences)

                formatted_fields = hypothesis(evidence=assembled_evidence, image=im, rules=rules,
                                              page_no=int(page_key.split('_')[-1]))

                with open(fields_json_folder + document_name + "_" + page_key + ".json", "w") as evfile:
                    json.dump(formatted_fields, evfile)

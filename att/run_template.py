import re
import io
from dateutil.parser import parse
from PIL import Image, ImageSequence
from pdfplumber.pdf import PDF
from pdf2image import convert_from_bytes
import numpy as np
from IDP_pipeline.ocr.utils import cvutils
from IDP_pipeline.ocr.ocr_entities.rzt_ocr.text_patch_detector import RZTTextPatchDetector
from IDP_pipeline.ocr.ocr_entities.tesseract.tesseract_ocr_entity import Tesseract_OCR

from IDP_pipeline.runner.rzt.Assemble_Evidences import assemble_evidences
from IDP_pipeline.hypotheses.ocr_pattern_hypothesis.RND.Bordered_table import Bordered_Table
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

island_alg = "CONNECTED_COMPONENT"
# island_alg= "CONTOUR"
# island_alg ="HISTOGRAM"
root_folder = "/home/rztuser/IDP/run_result/"
image_folder = "/home/rztuser/IDP/images/"
rule_json = "/home/rztuser/IDP/Jsons/att_rules_new1.json"
run_evidence = False
required_evidences = ["tesseract", "rzt_ocr"]


def tesseract_evidence(img):
    # tesseract_words, tesseract_content = Tesseract_OCR.get_pyocr_words_strings(text_patch=img)
    # res = {
    #    'tesseract_words': tesseract_words,
    #   'tesseract_content': tesseract_content
    # }
    # return res
    site = 'http://192.168.60.45:5000'
    _, img_encoded = cv2.imencode('.jpg', tp)
    response = requests.post(site + '/tesseract_evidence', data=img_encoded.tostring())
    tesseract_json = response.json()
    return tesseract_json


def write_text_on_image(img, word_patch_with_string, color=(0, 0, 0), fx=2, fy=2):
    try:
        word_patch_with_string = {tuple(eval(i)): j for i, j in word_patch_with_string.items()}
    except Exception as e:
        pass
    new_img = np.zeros(img.shape) + 255
    new_img = cv2.resize(new_img, (0, 0), fx=fx, fy=fy)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for patch, text in word_patch_with_string.items():
        cv2.putText(new_img, text, (int(patch[1] * fx), int(patch[2] * fx)), font, 1, color, 2, cv2.LINE_AA)
    return new_img


def hypothesis(evidence, image, rules):
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
    # print(images_folder + file_name + ".jpg")
    imgs = [image]
    # print(imgs)
    # exit()

    # Get word patches!
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

    final_data = []
    for page, img in zip(all_results, imgs):
        temp = []
        for frame in page:
            temp.append(
                [(frame.coordinates[0], frame.coordinates[1]), frame.name, Word.join(*frame.contains['words']),
                 frame.confidence])

        final_data.append(temp)
    return (final_data)


def tiff_to_jpg(binary_data: str):
    im = Image.open(io.BytesIO(binary_data))
    im_arr = []
    for i, page in enumerate(ImageSequence.Iterator(im)):
        im_arr.append(np.asarray(page.convert('RGB')))
    return im_arr


def fetch_words(binary_input, filepath):
    def get_text_patch_from_image(image):
        if island_alg == "CONNECTED_COMPONENT":
            text_patch_coords = RZTTextPatchDetector(page_image=image).detect_using_connected_components()
        elif island_alg == "CONTOUR":
            text_patch_coords = RZTTextPatchDetector(page_image=image).detect_using_contours()
        else:
            text_patch_coords = RZTTextPatchDetector(page_image=image).detect_using_histogram()

        text_patches = [image[t[1]:t[3], t[0]:t[2]].copy() for t in text_patch_coords]

        text_patch_list = []
        for e1, tp in enumerate(text_patches):
            if tp is None or tp.shape[0] == 0 or tp.shape[1] == 0:
                continue
            orientation = cvutils.get_text_orientation_using_word_width(tp)
            text = text_patch_coords[e1]
            if orientation == "VERTICAL":
                text_patch_key = str(text) + "_V"
            else:
                text_patch_key = str(text) + "_H"
            text_patch_list.append(text_patch_key)

        return text_patch_list

    document_name = filepath.split('/')[-1]
    document_name_without_ext = document_name.rsplit('.', 1)[0]
    extension = document_name.split('.')[-1].lower()
    result_dict = {}
    if extension == 'tif' or extension == 'tiff':
        images = tiff_to_jpg(binary_input)

        for e, image in enumerate(images):
            page_filename = document_name_without_ext + "_page_" + str(e)
            text_patch_list = get_text_patch_from_image(image, page_filename)
            result_dict['page_' + str(e)] = {
                'words': [],
                'text_images': text_patch_list,
                'numpy_image': image
            }
    elif extension == 'pdf':
        pdf = PDF(io.BytesIO(binary_input))
        images = convert_from_bytes(binary_input)
        for e, page in enumerate(pdf.pages):
            image = images[page.page_number - 1]
            factor = float(image.size[0] / page.width)
            image = np.asarray(image)
            text_patch_list = []
            image_area = image.shape[0] * image.shape[1]
            print("image shape", image.shape)
            try:
                is_scanned_image = False
                if "image" in page.objects:

                    if len(page.objects["image"]) == 1:
                        imageObject = page.objects["image"][0]
                        x0, x1, top, bottom = (round(float(imageObject[k]) * factor) for k in
                                               ('x0', 'x1', 'top', 'bottom'))

                        if (top, x0) == (0, 0) and abs(bottom - image.shape[0]) < 10 and abs(x1 - image.shape[1]) < 10:
                            is_scanned_image = True
                    else:
                        for imageObject in page.objects["image"]:
                            x0, x1, top, bottom = (round(float(imageObject[k]) * factor) for k in
                                                   ('x0', 'x1', 'top', 'bottom'))

                            tp = image[top:bottom, x0:x1].copy()
                            if ((bottom - top) * (x1 - x0)) >= 0.5 * image_area:
                                # bigger image patch. Try detecting islands
                                text_patch_list.extend(get_text_patch_from_image(tp))
                            else:
                                # smaller image patches save as a single text patch
                                orientation = cvutils.get_text_orientation_using_word_width(tp)

                                text = [x0, top, x1, bottom]
                                if orientation == "VERTICAL":
                                    text_patch_key = str(text) + "_V"
                                else:
                                    text_patch_key = str(text) + "_H"
                                text_patch_list.append(text_patch_key)
                patches = []
                if not is_scanned_image:
                    for patch in page.extract_words():
                        x0, x1, top, bottom = (round(float(patch[k]) * factor) for k in
                                               ('x0', 'x1', 'top', 'bottom'))
                        patches.append([[top, x0, bottom, x1], patch['text']])

                if len(patches) == 0:
                    text_patch_list = get_text_patch_from_image(image)

                result_dict['page_' + str(e)] = {
                    'words': patches,
                    'text_images': text_patch_list,
                    'numpy_image': image
                }
            except OverflowError as err:
                text_patch_list = get_text_patch_from_image(image)
                result_dict['page_' + str(e)] = {
                    'words': [],
                    'text_images': text_patch_list,
                    'numpy_image': image
                }
    else:
        im = Image.open(io.BytesIO(binary_input))
        image = np.asarray(im)
        text_patch_list = get_text_patch_from_image(image)
        result_dict['page_0'] = {
            'words': [],
            'text_images': text_patch_list,
            'numpy_image': image
        }

    return result_dict, document_name


def fetch_image_lines(image: np.ndarray):
    lines = cvutils.detect_lines(image)
    lines_dict = {"horizontal_lines": [], "vertical_lines": []}
    for line in lines:
        if (line[1] - line[3]) < (line[2] - line[0]):
            lines_dict["horizontal_lines"].append([line[3], line[0], line[1], line[2]])
        else:
            lines_dict["vertical_lines"].append([line[3], line[0], line[1], line[2]])
    return lines_dict


def fetch_image_bordered_tables(image: np.ndarray):
    coor_list = Bordered_Table(image).get_coordinate_list()
    return coor_list


import cv2
import imutils
from IDP_pipeline.ocr.ocr_entities.rzt_ocr.rzt_ocr_entity import RZT_OCR
from IDP_pipeline.ocr.ocr_entities.rzt_ocr.word_detector import RZTWordDetector
import os
import tensorflow as tf
import requests

from IDP_pipeline.ocr.ocr_entities.rzt_ocr.lstm_prediction_model import PredictionModel as PredictionModel
from IDP_pipeline.ocr.ocr_entities.google_cloud_vision.gv_ocr_entity import Google_Cloud_Vision_OCR
import json
import glob


def draw_patches_and_write_image(im, text_patch_list, evidences, word_list_key, file_path):
    img_with_patch = deepcopy(im)
    for text_patch_key in text_patch_list:
        t = eval(text_patch_key.split("_")[0])
        orientation = text_patch_key.split("_")[1]
        tp = im[t[1]:t[3], t[0]:t[2]].copy()

        image_gray = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
        image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if orientation == "V":
            tp = imutils.rotate_bound(tp, 90)

        for evidence_word in evidences[text_patch_key][word_list_key]:
            print("evidence word", evidence_word)
            word = evidence_word[0]
            word_ = [word[0] + t[1], word[1] + t[0], word[2] + t[1], word[3] + t[0]]
            cv2.rectangle(img_with_patch, (t[0], t[1]), (t[2], t[3]), (255, 0, 0), 2)
            # cv2.rectangle(img_with_patch, (word_[1], word_[0]), (word_[3], word_[2]), (0, 0, 255), 2)
    cv2.imwrite(file_path, img_with_patch)


if __name__ == "__main__":

    rules = json.load(open(rule_json))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.2
    session = tf.InteractiveSession(config=config_sess)
    prediction_model = PredictionModel(
        model_dir=None,
        session=session)

    # site = 'http://192.168.60.45:5000'
    # filepath = "/Users/sunilkumar/ocr/al_data/docs1/RW00275473_2017-12-01.pdf"

    evidence_folder = root_folder + "evidence/"
    assembled_image_folder = root_folder + "assembled_image/"
    predicted_text_folder = root_folder + "predicted_text/"
    rzt_image_folder = root_folder + "rzt_image/"
    tesseract_image_folder = root_folder + "tesseract_image/"
    gv_image_folder = root_folder + "gv_image/"
    text_image_folder = root_folder + "text_image_folder/"
    fields_json_folder = root_folder + "fields_json/"
    fields_image_folder = root_folder + "fields_image/"
    xml_result = root_folder + "xml_output/"

    if not os.path.isdir(fields_json_folder):
        os.mkdir(fields_json_folder)

    if not os.path.isdir(predicted_text_folder):
        os.mkdir(predicted_text_folder)

    if not os.path.isdir(fields_image_folder):
        os.mkdir(fields_image_folder)

    if not os.path.isdir(assembled_image_folder):
        os.mkdir(assembled_image_folder)

    if not os.path.isdir(tesseract_image_folder):
        os.mkdir(tesseract_image_folder)

    if not os.path.isdir(gv_image_folder):
        os.mkdir(gv_image_folder)

    if not os.path.isdir(rzt_image_folder):
        os.mkdir(rzt_image_folder)

    if not os.path.isdir(evidence_folder):
        os.mkdir(evidence_folder)

    if not os.path.isdir(text_image_folder):
        os.mkdir(text_image_folder)

    if not os.path.isdir(xml_result):
        os.mkdir(xml_result)

    address_fields = ['address', 'city', 'state', 'vendor_id', 'zip']
    invoice_header = ['account', 'amount_due', 'date', 'due_date', 'number']
    invoice_info = ['total_current_charges', 'late_charges', 'PDB']
    state_list = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","AS","DC","FM","GU","MH","MP","PW","PR","VI"]
    all_invoice_dict = dict()
    invoice_details = dict()
    elements_count = dict()
    confidence_score_map = dict()
    for filepath in glob.glob(image_folder + "*"):
        # for filepath in glob.glob("/Users/sunilkumar/ocr/Table_Data/highpeak/*.pdf"):
        invoice_dict = dict()
        invoice_dict['invoice_info'] = dict()
        invoice_dict['invoice_info']['charge'] = list()
        invoice_dict['invoice_details'] = dict()

        with open(filepath, "rb") as binfile:
            pdf_words, document_name = fetch_words(binfile.read(), filepath)

            print("Fetched words", filepath, document_name)
            invoice_details[filepath] = dict()
            elements_count[filepath] = dict()
            for page_key, val in pdf_words.items():
                page_file = document_name + "_" + page_key + ".json"
                print(document_name, page_key)
                im = val["numpy_image"]
                invoice_details[filepath][page_key] = list()
                elements_count[filepath][page_key] = dict()
                if not run_evidence and os.path.isfile(evidence_folder + page_file):
                    print("Loading evidence from ", evidence_folder + page_file)
                    with open(evidence_folder + page_file, "r") as evfile:
                        assembled_evidence = json.load(evfile)
                else:
                    print("Running evidence")
                    text_im = deepcopy(im)
                    evidence_list = []
                    text_patch_list = val["text_images"]
                    if "rzt_ocr" in required_evidences:
                        rzt_evidences = {}
                        for text_patch_key in text_patch_list:
                            print(text_patch_key)
                            t = eval(text_patch_key.split("_")[0])
                            orientation = text_patch_key.split("_")[1]
                            tp = im[t[1]:t[3], t[0]:t[2]].copy()
                            color_ = (0, 0, 255)
                            if orientation == "V":
                                color_ = (255, 0, 0)
                            cv2.rectangle(text_im, (t[0], t[1]), (t[2], t[3]), color_, 2)

                            image_gray = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
                            image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                            if cvutils.is_inverted_text_patch(image_bin):
                                tp = (255 - tp).astype(np.uint8)

                            if orientation == "V":
                                tp = imutils.rotate_bound(tp, 90)

                            rzt_evidence = RZT_OCR(image=tp.copy(), lstm_model=prediction_model,
                                                   word_detector=RZTWordDetector.get_words_using_word_space_cluster).get_word_coordinates_with_string()
                            print(rzt_evidence)

                            rzt_evidences[text_patch_key] = rzt_evidence
                        # draw and write rzt images
                        image_file = rzt_image_folder + document_name + "_" + page_key + ".jpg"
                        draw_patches_and_write_image(im, text_patch_list, rzt_evidences, "word_list", image_file)
                        evidence_list.append(rzt_evidences)

                    cv2.imwrite(text_image_folder + document_name + "_" + page_key + ".jpg", text_im)

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

                            # _, img_encoded = cv2.imencode('.jpg', tp)
                            # response = requests.post(site + '/tesseract_evidence', data=img_encoded.tostring())
                            # tesseract_json = response.json()

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
                            # cv2.imwrite("/Users/sunilkumar/ocr/al_data/text_patches_r/" + text_patch_key + ".jpg",tp)

                            gv_evidence = Google_Cloud_Vision_OCR(image=tp).get_word_coordinate_with_string()

                            gv_evidences[text_patch_key] = gv_evidence
                        evidence_list.append(gv_evidences)

                    assembled_evidence = assemble_evidences(im, evidence_list, val["words"], text_patch_list,
                                                            required_evidences)

                # assembled_evidence["lines"] = fetch_image_lines(im)
                # assembled_evidence["tables"] = fetch_image_bordered_tables(im)
                print(page_key, assembled_evidence)
                fields = hypothesis(evidence=assembled_evidence, image=im, rules=rules)
                field_im = deepcopy(im)

                for page_result in fields:
                    full_address = ''
                    for a in page_result:
                        if a[1] == 'full_address':
                            if ('full_address' in confidence_score_map) and (confidence_score_map['full_address'] > a[3]):
                                pass
                            else:
                                full_address = a[2]
                                confidence_score_map[a[1]] = a[3]
                    for field in page_result:
                        if field[1] == 'full_address':
                            cv2.rectangle(field_im, field[0][0], field[0][1], (0, 0, 255), 2)
                            continue
                        if (field[1] in confidence_score_map) and (confidence_score_map[field[1]] > field[3]):
                            continue
                        else:
                            confidence_score_map[field[1]] = field[3]
                        if field[1] in address_fields:
                            if (full_address is not None) and (field[2] in full_address):
                                invoice_dict[field[1]] = field[2]
                                if (field[1] == 'state') and (field[2] in state_list):
                                    invoice_dict['country'] = 'United States'
                                    invoice_dict['currency'] = 'USD'
                            elif field[1] not in invoice_dict:
                                invoice_dict[field[1]] = field[2]
                                if (field[1] == 'state') and (field[2] in state_list):
                                    invoice_dict['country'] = 'United States'
                                    invoice_dict['currency'] = 'USD'
                        elif field[1] in invoice_header:
                            if field[1] == 'account':
                                field[2] = re.sub('[ -]', '', field[2])
                            if field[1] == 'amount_due':
                                field[2] = re.sub('[ $]', '', field[2])
                                if field[2].startswith('.'):
                                    field[2] = '0' + field[2]
                                if field[2].startswith('$.'):
                                    field[2] = re.sub('[$]', '', field[2])
                                    field[2] = '0' + field[2]
                                if 'CR' in field[2]:
                                    field[2] = '-' + field[2].replace('CR', '')
                            if (field[1] == 'due_date') or (field[1] == 'date'):
                                field[2] = parse(field[2]).strftime('%m/%d/%Y')
                            invoice_dict[field[1]] = field[2]
                        elif field[1] in invoice_info:
                            if field[2].startswith('.'):
                                field[2] = '0' + field[2]
                            if field[2].startswith('$.'):
                                field[2] = re.sub('[$]', '', field[2])
                                field[2] = '0' + field[2]
                            if 'CR' in field[2]:
                                field[2] = '-' + field[2].replace('CR', '')
                            invoice_dict['invoice_info']['charge'].append({'amount': field[2], 'type': field[1]})
                        else:
                            if field[2].startswith('.'):
                                field[2] = '0' + field[2]
                            if field[2].startswith('$.'):
                                field[2] = re.sub('[$]', '', field[2])
                                field[2] = '0' + field[2]
                            if 'CR' in field[2]:
                                field[2] = '-' + field[2].replace('CR', '')
                            if ('Billed' == field[1]) or ('Charges' == field[1]):
                                field[2] = re.sub('[\\_\\W]', '', field[2])
                            invoice_details[filepath][page_key].append([field[0][0][1], field[1], field[2]])
                            if field[1] in elements_count[filepath][page_key]:
                                elements_count[filepath][page_key][field[1]] += 1
                            else:
                                elements_count[filepath][page_key][field[1]] = 1
                        cv2.rectangle(field_im, field[0][0], field[0][1], (0, 0, 255), 2)
                cv2.imwrite(fields_image_folder + document_name + "_" + page_key + ".jpg", field_im)

                with open(fields_json_folder + document_name + "_" + page_key + ".json", "w") as evfile:
                    json.dump(fields, evfile)

                color_ = (255, 0, 0)
                thickness = 1
                with open(evidence_folder + page_file, "w") as evfile:
                    json.dump(assembled_evidence, evfile)
                word_patch_with_string = dict()
                for word_key, word in assembled_evidence["evidence_words"].items():
                    color_ = (255, 0, 0)
                    if word["true_pdf"] == 1:
                        color_ = (255, 0, 0)
                    else:
                        color_ = (0, 0, 255)

                    thickness = 1
                    # if word["stroke_width"] > 2.9:
                    #     thickness= 3

                    w = word["assembled_result"][0]
                    word_patch_with_string[tuple(w)] = word["assembled_result"][1]
                    cv2.rectangle(im, (w[1], w[0]), (w[3], w[2]), color_, thickness)
                cv2.imwrite(assembled_image_folder + document_name + "_" + page_key + ".jpg", im)
                image_with_text = write_text_on_image(deepcopy(im), word_patch_with_string)
                cv2.imwrite(predicted_text_folder + document_name + "_" + page_key + ".jpg", image_with_text)
            all_invoice_dict[filepath] = invoice_dict

    for document in invoice_details:
        pdb_subtract = True
        invoice_dict = all_invoice_dict[document]
        invoice_dict['invoice_details'][''] = dict()
        invoice_dict['invoice_details']['']['charge'] = list()
        for page in invoice_details[document]:
            item_no = None
            sorted_elements = sorted(invoice_details[document][page])
            print(elements_count[document][page])
            late_charges = 0
            if pdb_subtract:
                for info in invoice_dict['invoice_info']['charge']:
                    if 'late_charges' == info['type']:
                        late_charges = info['amount']
                        break
            if pdb_subtract:
                for info in invoice_dict['invoice_info']['charge']:
                    if 'PDB' == info['type']:
                        info['amount'] = str(round((float(info['amount']) - float(late_charges)), 2))
                        pdb_subtract = False
                        break
            for element in sorted_elements:
                print(element)
                if (element[1] == 'Billed') or (element[1] == 'Charges'):
                    if bool(re.search(r'\d', element[2])):
                        item_no = element[2]
                    continue
                if item_no is None:
                    invoice_dict['invoice_details']['']['charge'].append({'description': element[1],
                                                                          'amount': element[2]})
                    continue
                if item_no not in invoice_dict['invoice_details']:
                    invoice_dict['invoice_details'][item_no] = dict()
                    invoice_dict['invoice_details'][item_no]['charge'] = list()
                    invoice_dict['invoice_details'][item_no]['charge'].append({'description': element[1],
                                                                               'amount': element[2]})
                else:
                    invoice_dict['invoice_details'][item_no]['charge'].append({'description': element[1],
                                                                               'amount': element[2]})
        print(json.dumps(invoice_dict, indent=4))
        from src.json2xml import Json2xml

        data = Json2xml.fromstring(json.dumps(invoice_dict)).data
        data_object = Json2xml(data)
        xml_output = data_object.json2xml()
        print('\n\n')
        print('XML Result-------------------------------------------------------')
        print(xml_output)  # xml output
        doc_name = os.path.split(document)[1].split('.')[0]
        with open(xml_result + doc_name + ".xml", "w") as evfile:
            evfile.write(xml_output)
        evfile.close()
    print(all_invoice_dict)

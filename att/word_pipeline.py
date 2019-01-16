import io
from PIL import Image, ImageSequence
from pdfplumber.pdf import PDF
from pdf2image import convert_from_bytes
import numpy as np
from IDP_pipeline.ocr.utils import cvutils
from IDP_pipeline.ocr.ocr_entities.rzt_ocr.text_patch_detector import RZTTextPatchDetector
from IDP_pipeline.runner.rzt.Assemble_Evidences import assemble_evidences
from IDP_pipeline.hypotheses.ocr_pattern_hypothesis.RND.Bordered_table import Bordered_Table
from copy import deepcopy

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

island_alg = "CONNECTED_COMPONENT"
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


if __name__ == "__main__":

    root_folder = "/home/rztuser/IDP/test/"
    image_path = "/home/rztuser/IDP/images/ALSAC_ATT_30968551013871_20180610_309685510106.pdf"
    run_evidence = False

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.2
    session = tf.InteractiveSession(config=config_sess)
    prediction_model = PredictionModel(
        model_dir=None,
        session=session)

    evidence_folder = root_folder + "evidence/"
    text_image_folder = root_folder + "text_image_folder/"

    if not os.path.isdir(evidence_folder):
        os.mkdir(evidence_folder)

    if not os.path.isdir(text_image_folder):
        os.mkdir(text_image_folder)

    with open(image_path, "rb") as binfile:
        pdf_words, document_name = fetch_words(binfile.read(), image_path)

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

                cv2.imwrite(text_image_folder + document_name + "_" + page_key + ".jpg", text_im)
                assembled_evidence = assemble_evidences(im, evidence_list, val["words"], text_patch_list,
                                                        required_evidences)

            with open(evidence_folder + page_file, "w") as evfile:
                json.dump(assembled_evidence, evfile)

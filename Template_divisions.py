import tensorflow as tf
import numpy as np
import cv2
import os
import json
import pandas as pd
from bson.objectid import ObjectId
from six.moves import cPickle as pickle  # for performance
# from sklearn.metrics.classification import confusion_matrix
from pymongo import MongoClient
from bson import binary
import pickle
from pdf2image import convert_from_path

CONNECTION_STRING = "mongodb://localhost:27017/?readPreference=primary"
DATABASE = "tangoe-idp"
COLLECTION_NAME = "pdfs"
MODEL_PATH = "/home/rztuser/IDP/tabletemplates/SimNet/"

"""
Connect to Database
"""


def connect_to_db():
    client = MongoClient(CONNECTION_STRING)
    return client


def close_connection(client):
    client.close()


def add_new_template(template):
    client = connect_to_db()
    x = client[DATABASE][COLLECTION_NAME].insert_one(template)
    print(x.inserted_id)
    client.close()


def get_ops():
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # /Users/kevin/Downloads/SimNet
    new_saver = tf.train.import_meta_graph(MODEL_PATH + 'sim_model_285000.ckpt.meta')
    new_saver.restore(sess, save_path=MODEL_PATH + 'sim_model_285000.ckpt')

    op_dict = {
        "input1": tf.get_collection('input1')[0],
        "input2": tf.get_collection('input2')[0],
        "net1": tf.get_collection('net1')[0],
        "net2": tf.get_collection('net2')[0],
        "train_bn": tf.get_collection('train_bn')[0]
    }

    return sess, op_dict


"""
Returns the detected template
"""


def get_img_template(possible_templates):
    sorted_dist = sorted(possible_templates, key=lambda x: x[1])
    return sorted_dist[0][0]


"""
Finding new centroids
"""


def update_template_dict(detected_template, feature_array, img_path):
    # template = {}
    detected_templated_vector = detected_template["templateVector"]
    detected_template_array = pickle.loads(detected_template["templateVector"])
    updated_template_array = ((detected_template_array * detected_template["countOfDocs"]) + feature_array)
    updated_template_array = updated_template_array / (detected_template["countOfDocs"] + 1)

    template_vector = binary.Binary(pickle.dumps(updated_template_array, protocol=len(updated_template_array.shape)))
    count_of_docs = detected_template["countOfDocs"] + 1.0
    docs = detected_template["Docs"].append(img_path[1])

    client = connect_to_db()

    res = client[DATABASE][COLLECTION_NAME].update_one({"Name": detected_template["Name"]}, {'$set':
        {
            "templateVector": template_vector,
            "countOfDocs": count_of_docs,
            "Docs":
                detected_template[
                    "Docs"]}
    }, upsert=True)

    print("updated result is ", res)

    close_connection(client)

    return
    # return new_feature_array


"""
Actual method
"""


def get_templates(img_path, all_templates, template_id="template_id_", dist_threshold=4.0,
                  save_path="/home/kartik/Documents/IDP-files/templates/"):
    print("GOING TO GET SESSION")
    sess, op_dict = get_ops()
    print("SESSION LOADED")

    possible_template_matches = []

    original_img = cv2.imread(img_path[0] + img_path[1] + ".jpg")
    im1 = cv2.resize(original_img, (224, 224))
    feature_array = sess.run(op_dict["net1"], feed_dict={op_dict["input1"]: [im1], op_dict['train_bn']: False})

    """
    Getting matches
    """
    for template in all_templates:
        template_vector = pickle.loads(template["templateVector"])
        dist = np.linalg.norm(template_vector - feature_array)
        if dist < dist_threshold:
            possible_template_matches.append([template, dist])

    # for id,feature in template_dict.items():
    #     dist = np.linalg.norm(feature-feature_array)
    #     if dist<dist_threshold:
    #         distance_list.append([id,dist])

    """
    For new template
    """
    if len(possible_template_matches) == 0:
        print("length is zero")
        template = {}
        # for now saving the image name , but later you can save something else
        template["Name"] = img_path[1]
        template["Organisations"] = []
        # for now saving the image name , but later save the actual doc ids
        template["Docs"] = [img_path[1]]
        template["countOfDocs"] = 1.0

        print("shape of feature array is ", feature_array.shape)
        feature_array = binary.Binary(pickle.dumps(feature_array, protocol=len(feature_array.shape)))
        template["templateVector"] = feature_array
        add_new_template(template)

        # template_dict[template_name]=feature_array
        # template_Image_dict[template_name]=[img_path[1]]

        # temp code for debugging...comment it later
        template_name = str(ObjectId())
        # os.mkdir(save_path + template_name)
        # cv2.imwrite(save_path+template_name+"/"+img_path[1]+".jpg",original_img)
        # print("New Template : ",template_name)
        return

    else:
        detected_template = get_img_template(possible_template_matches)
        # print("detected template id is ", str(detected_template_id))
        update_template_dict(detected_template, feature_array, img_path)
        return

        # template_dict[img_insert_template_id]=update_template_dict(template_dict[img_insert_template_id],feature_array,len(template_Image_dict[img_insert_template_id]))
        # template_Image_dict[img_insert_template_id].append(img_path[1])
        # cv2.imwrite(save_path + str(detected_template_id) + "/" + img_path[1] + ".jpg", original_img)
        # print("image saved")

        # print("Added to template : ",img_insert_template_id)
        # return template_dict,template_Image_dict


"""
gets all the existing templates from the database
"""


def get_all_templates():
    client = connect_to_db()
    try:
        templates = list(client[DATABASE][COLLECTION_NAME].find({}))
    except Exception as ex:
        print(ex)
    close_connection(client)
    return templates


"""
Initiation
"""


def template_clustering(img_folder_path):
    img_path = [img_folder_path, "0000"]

    filenames = os.listdir(img_path[0])
    errored = []

    """
    To run each image in detection
    """
    for enum, filename in enumerate(filenames):
        try:
            pdf_ext = '.pdf'
            if os.path.isdir(img_folder_path + '/' + filename):
                continue
            if filename.endswith(pdf_ext):
                if not os.path.exists(image_folder + 'images/'):
                    os.mkdir(image_folder + 'images/')
                jpg_file = filename.replace(pdf_ext, '.jpg')
                if not os.path.exists(image_folder + 'images/' + jpg_file):
                    pages = convert_from_path(image_folder + filename)
                    pages[0].save(image_folder + 'images/' + jpg_file)
                img_path[0] = img_folder_path + 'images/'
                filename = jpg_file
            else:
                img_path[0] = img_folder_path
            all_templates = get_all_templates()
            print("all templates are ", all_templates)
            if filename.startswith("."):
                continue
            print("\nIMG NO :", enum, "     IMG :", filename)
            file_split = filename.split(".")
            img_path[1] = '.'.join(file_split[0: len(file_split) - 1])
            get_templates(img_path, all_templates)
        except Exception as e:
            print("issue with This file", e)
            errored.append(filename)

    # """
    # Saving new data to json
    # """
    # for k,v in template_dict.items():
    #     template_dict[k]=v.tolist()
    # with open(template_dict_path, "w") as save_file:
    #     json.dump(template_dict, save_file)

    # with open(template_Image_dict_path,"w") as save_file:
    #     json.dump(template_Image_dict, save_file)

    # print("ERRORED IMAGES",errored)


if __name__ == '__main__':
    image_folder = "/home/rztuser/IDP/PDFs/"
    client = connect_to_db()
    template_clustering(img_folder_path=image_folder)
    close_connection(client)

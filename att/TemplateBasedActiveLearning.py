import tensorflow as tf
import uuid
import cv2
import pickle
from bson import binary
from pymongo import MongoClient
import numpy as np
import json
import sys
import os
import pdfplumber
from bson.objectid import ObjectId
from pdf2image import convert_from_path


class Singleton(type):
    """
    @created on: 2/4/19,
    @author: Kevin Xavier,
    Description: classes which interit from this class will have Singleton Properties

    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Loader(metaclass=Singleton):
    """
    @created on: 2/4/19,
    @author: Kevin Xavier,
    Description: Class used to load all the static informations and DB connection and configurations

    """
    MODEL_PATH = os.getenv("HOME") + "/IDP/scripts" + "/SimNet/"
    CONNECTION_STRING = "mongodb://localhost:27017/?readPreference=primary"
    DATABASE = "tangoe"
    TEMPLATE_COLLECTION_NAME = "template"
    IMAGE_COLLECTION_NAME = "image_collection"
    DOCUMENT_COLLECTION_NAME = "documents"
    # TEMPLATE_FOLDER = "/Users/kevin/Downloads/AL/"
    FILES_FOLDER = os.getenv("HOME") + "/IDP/results/"

    TEMPLATE_FOLDER = os.getenv("HOME") + "/IDP/Templates/"
    if not os.path.isdir(TEMPLATE_FOLDER):
        os.mkdir(TEMPLATE_FOLDER)

    # print("howmany")

    def __init__(self):
        """
            @created on: 2/4/19,
            @author: Kevin Xavier,
            Description: Definition to load Template Model from filesystem on constructor call itself.

        """
        # Loading model and session
        tf.reset_default_graph()
        print("Loaded_model")
        self.model_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.model_new_saver = tf.train.import_meta_graph(Loader.MODEL_PATH + 'sim_model_285000.ckpt.meta')
        self.model_new_saver.restore(self.model_sess, save_path=Loader.MODEL_PATH + 'sim_model_285000.ckpt')
        self.model_op_dict = {
            "input1": tf.get_collection('input1')[0],
            "input2": tf.get_collection('input2')[0],
            "net1": tf.get_collection('net1')[0],
            "net2": tf.get_collection('net2')[0],
            "train_bn": tf.get_collection('train_bn')[0]
        }
        print("MODEL SESSION LOADED")

    @staticmethod
    def connect_to_db():
        client = MongoClient(Loader.CONNECTION_STRING)
        return client

    @staticmethod
    def close_connection(client):
        client.close()


class TemplateBasedActiveLearning():
    TEMPLATE_THRESHOLD_FOR_DIVISION = 6
    TEMPLATE_THRESHOLD_FOR_ALGORITHM_REVIEW = 1

    def __init__(self, file_id, image, organisation):
        """
            @created on: 2/4/19,
            @author: Kevin Xavier,
            Description: Sets all the attributes of image object and initiates the initial run.

        """
        self.file_id = file_id
        self.image = image
        self.organisation = organisation
        self.status = "pending"
        self.template_id = "notAssigned"
        self.possible_template_ids = []
        self.binary_feature_array = bytes()
        self.run()

    @staticmethod
    def get_template_scores(image):
        im1 = cv2.resize(image, (224, 224))
        # self.resized_image=im1
        feature_array = Loader().model_sess.run(Loader().model_op_dict["net1"],
                                                feed_dict={Loader().model_op_dict["input1"]: [im1],
                                                           Loader().model_op_dict['train_bn']: False})
        return feature_array

    @staticmethod
    def convert_to_binary(arr):
        binary_arr = binary.Binary(pickle.dumps(arr,
                                                protocol=len(arr.shape)))
        return binary_arr

    @staticmethod
    def convert_from_binary(arr):
        obj = pickle.loads(arr)
        return obj

    @staticmethod
    def convert_to_blur_image(image):
        blur_image = cv2.blur(image,(25,25))
        return blur_image

    @staticmethod
    def create_new_template_structure(file_id, image, organisation, feature_array, rule_json=None, t_id=None):

        template = {}
        if t_id == None:
            template["templateId"] = str(uuid.uuid4())
        else:
            template["templateId"] = t_id

        # self.template_id = template["id"]
        if rule_json == None:
            rule_json = {}
        template["organisations"] = {organisation: {
            "ruleJson": rule_json,
            "ruleJsonVersions": [rule_json]
        }
        }
        template["docIds"] = [file_id]
        # template["docIds"]=[self.file_id]
        template["countOfDocs"] = 1.0
        # feature_array=TemplateBasedActiveLearning.get_template_scores(image)
        binary_feature_array = TemplateBasedActiveLearning.convert_to_binary(feature_array)
        template["binaryTemplateVector"] = binary_feature_array
        template["closestBinaryTemplateVector"] = binary_feature_array
        blur_image = TemplateBasedActiveLearning.convert_to_blur_image(image)
        cv2.imwrite(Loader.TEMPLATE_FOLDER + template["templateId"] + ".jpg", blur_image)
        template["closestImagePath"] = Loader.TEMPLATE_FOLDER + template["templateId"] + ".jpg"
        template["toBeReviewed"] = []
        return template

        # # client = self.client.connect_to_db()
        # # print(Loader.DATABASE)
        # # print(Loader.COLLECTION_NAME)
        # # template={"aab":"asd"}
        # x = self.client[Loader.DATABASE][Loader.COLLECTION_NAME].insert_one(template)
        # self.status = "AlgorithmGenerated"

    @staticmethod
    def get_all_templates(client):
        templates = []
        # client = ActiveLearning.connect_to_db()
        try:
            templates = list(client[Loader.DATABASE][Loader.TEMPLATE_COLLECTION_NAME].find({}))
        except Exception as ex:
            print(ex)
        # ActiveLearning.close_connection(client)
        return templates

    @staticmethod
    def get_closest_templates(template_list, target_featurearray):
        possible_template_matches = []
        for template in template_list:
            template_vector = pickle.loads(template["binaryTemplateVector"])
            dist = np.linalg.norm(template_vector - target_featurearray)
            # print(template["id"])
            # print(dist)
            if dist < TemplateBasedActiveLearning.TEMPLATE_THRESHOLD_FOR_DIVISION:
                possible_template_matches.append({"id": template["templateId"], "distance": dist})  # only pass id.
        possible_template_matches = sorted(possible_template_matches,
                                           key=lambda x: x["distance"])
        return possible_template_matches

    @staticmethod
    def insert_record_to_DB(client, collection_name, record):
        x = client[Loader.DATABASE][collection_name].insert_one(record)

    @staticmethod
    def update_template_record_to_DB(client, collection_name, record):
        res = client[Loader.DATABASE][collection_name].update_one({"templateId": record["templateId"]},
                                                                  {'$set': record},
                                                                  upsert=True)

    @staticmethod
    def update_image_record_to_DB(client, collection_name, record):
        print(record)
        print(collection_name)
        res = client[Loader.DATABASE][collection_name].update_one({"fileId": record["fileId"]},
                                                                  {'$set': record},
                                                                  upsert=True)

    @staticmethod
    def update_doc_record_to_DB(client, collection_name, record):
        print(record)
        print(collection_name)
        res = client[Loader.DATABASE][collection_name].update_one({"_id": record["_id"]},
                                                                  {'$set': record},
                                                                  upsert=True)

    @staticmethod
    def resize_image(image):
        resized = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        return resized

    @staticmethod
    def pull_template_from_DB(template_id, client):
        pulled_template = dict(
            list(client[Loader.DATABASE][Loader.TEMPLATE_COLLECTION_NAME].find({"templateId": template_id}))[0])
        return pulled_template

    @staticmethod
    def pull_image_from_DB(file_id, client):
        print("ffff", file_id)
        pulled_image = dict(
            list(client[Loader.DATABASE][Loader.IMAGE_COLLECTION_NAME].find({"fileId": file_id}))[0])
        return pulled_image

    @staticmethod
    def pull_document_from_DB(file_id, client):
        pulled_doc = dict(
            list(client[Loader.DATABASE][Loader.DOCUMENT_COLLECTION_NAME].find({"_id": ObjectId(file_id)}))[0])
        return pulled_doc

    @staticmethod
    def map_template_to_obj(template_structure, image_structure, status, distance_to_image, image, feedback=None):
        if status == "algorithmGenerated":
            image_structure["status"] = "algorithmGenerated"
            image_structure["templateId"] = template_structure["templateId"]
            # image_structure["possibleTemplateIds"] = [template_structure["templateId"]]
        elif status == "algorithmPredicted":
            if distance_to_image < TemplateBasedActiveLearning.TEMPLATE_THRESHOLD_FOR_ALGORITHM_REVIEW:
                image_structure["status"] = "algorithmReviewed"
                image_structure["templateId"] = template_structure["templateId"]
                # image_structure["possible templaes"] -> should go inside if mian

                # DB updations

                if image_structure["organisation"] not in template_structure["organisations"].keys():
                    random_org = dict(list(template_structure["organisations"].values())[0])
                    template_structure["organisations"][image_structure["organisation"]] = random_org
                template_structure["docIds"].append(image_structure["fileId"])
                template_structure["countOfDocs"] = template_structure["countOfDocs"] + 1.0

                updated_vector = ((TemplateBasedActiveLearning.convert_from_binary(
                    template_structure["binaryTemplateVector"]) * (template_structure[
                                                                       "countOfDocs"] - 1)) + TemplateBasedActiveLearning.convert_from_binary(
                    image_structure["binaryTemplateVector"])) / template_structure["countOfDocs"]

                # new_vector = ((pickle.loads(template_structure["binaryTemplateVector"]) * pulled_template[
                #     "countOfDocs"]) + self.feature_array) / pulled_template["countOfDocs"] + 1

                current_dist = np.linalg.norm(updated_vector - TemplateBasedActiveLearning.convert_from_binary(
                    image_structure["binaryTemplateVector"]))
                previous_dist = np.linalg.norm(updated_vector - TemplateBasedActiveLearning.convert_from_binary(
                    template_structure["closestBinaryTemplateVector"]))

                if current_dist < previous_dist:
                    template_structure["closestBinaryTemplateVector"] = image_structure["binaryTemplateVector"]
                    blur_image = TemplateBasedActiveLearning.convert_to_blur_image(image)
                    cv2.imwrite(Loader.TEMPLATE_FOLDER + template_structure["templateId"] + ".jpg", blur_image)
                    template_structure["closestImagePath"] = Loader.TEMPLATE_FOLDER + template_structure[
                        "templateId"] + ".jpg"
            else:
                image_structure["status"] = "algorithmPredicted"
                image_structure["templateId"] = template_structure["templateId"]
                if image_structure["organisation"] not in template_structure["organisations"].keys():
                    random_org = dict(list(template_structure["organisations"].values())[0])
                    template_structure["organisations"][image_structure["organisation"]] = random_org
                template_structure["toBeReviewed"].append(image_structure["fileId"])

        elif status == "userReviewed":

            # image_structure["status"] = "userReviewed"
            # image_structure["templateId"] = template_structure["templateId"]
            # image_structure["possible templaes"] -> should go inside if mian

            # DB updations
            print(feedback)
            if feedback["ruleModification"]:
                with open(Loader.FILES_FOLDER + "/" + image_structure["fileId"] + "/" + "hints.json") as f:
                    rule = json.load(f)
                template_structure["organisations"][image_structure["organisation"]]["ruleJson"] = rule
                template_structure["organisations"][image_structure["organisation"]]["ruleJsonVersions"].append(rule)

            if image_structure["status"] == "algorithmGenerated" or image_structure["status"] == "algorithmReviewed":
                return template_structure, image_structure

            image_structure["status"] = "userReviewed"


            # if feedback["ruleJson"]["newJson"] != "None":
            #     template_structure["organisations"][image_structure["organisation"]]["ruleJson"] = feedback["ruleJson"][
            #         "newJson"]
            #     template_structure["organisations"][image_structure["organisation"]]["ruleJsonVersions"].append(
            #         feedback["ruleJson"]["newJson"])

            # if image_structure["organisation"] not in template_structure["organisations"].keys():
            #     random_org = dict(list(template_structure["organisations"].values())[0])
            #     template_structure["organisations"][image_structure["organisation"]] = random_org
            template_structure["docIds"].append(image_structure["fileId"])
            template_structure["countOfDocs"] = template_structure["countOfDocs"] + 1.0

            updated_vector = ((TemplateBasedActiveLearning.convert_from_binary(
                template_structure["binaryTemplateVector"]) * (template_structure[
                                                                   "countOfDocs"] - 1)) + TemplateBasedActiveLearning.convert_from_binary(
                image_structure["binaryTemplateVector"])) / template_structure["countOfDocs"]

            # new_vector = ((pickle.loads(template_structure["binaryTemplateVector"]) * pulled_template[
            #     "countOfDocs"]) + self.feature_array) / pulled_template["countOfDocs"] + 1

            current_dist = np.linalg.norm(updated_vector - TemplateBasedActiveLearning.convert_from_binary(
                image_structure["binaryTemplateVector"]))
            previous_dist = np.linalg.norm(updated_vector - TemplateBasedActiveLearning.convert_from_binary(
                template_structure["closestBinaryTemplateVector"]))

            if current_dist < previous_dist:
                template_structure["closestBinaryTemplateVector"] = image_structure["binaryTemplateVector"]
                blur_image = TemplateBasedActiveLearning.convert_to_blur_image(image)
                cv2.imwrite(Loader.TEMPLATE_FOLDER + template_structure["templateId"] + ".jpg", blur_image)
                template_structure["closestImagePath"] = Loader.TEMPLATE_FOLDER + template_structure[
                    "templateId"] + ".jpg"
            print(image_structure["fileId"])
            template_structure["toBeReviewed"].remove(image_structure["fileId"])

        elif status == "userGenerated":
            image_structure["status"] = "userGenerated"
            image_structure["templateId"] = template_structure["templateId"]
        elif status == "userRecalibrated":
            image_structure["status"] = "userRecalibrated"
            # print("feeeed",feedback)
            if not feedback["ruleModification"]:
                with open(Loader.FILES_FOLDER + "/" + image_structure["fileId"] + "/" + "hints.json") as f:
                    rule = json.load(f)
                template_structure["organisations"][image_structure["organisation"]]["ruleJson"] = rule
                template_structure["organisations"][image_structure["organisation"]]["ruleJsonVersions"].append(rule)

            # if feedback["ruleJson"]["newJson"] != "None":
            #     template_structure["organisations"][image_structure["organisation"]]["ruleJson"] = feedback["ruleJson"][
            #         "newJson"]
            #     template_structure["organisations"][image_structure["organisation"]]["ruleJsonVersions"].append(
            #         feedback["ruleJson"]["newJson"])

            # if image_structure["organisation"] not in template_structure["organisations"].keys():
            #     random_org = dict(list(template_structure["organisations"].values())[0])
            #     template_structure["organisations"][image_structure["organisation"]] = random_org
            template_structure["docIds"].append(image_structure["fileId"])
            template_structure["countOfDocs"] = template_structure["countOfDocs"] + 1.0

            updated_vector = ((TemplateBasedActiveLearning.convert_from_binary(
                template_structure["binaryTemplateVector"]) * (template_structure[
                                                                   "countOfDocs"] - 1)) + TemplateBasedActiveLearning.convert_from_binary(
                image_structure["binaryTemplateVector"])) / template_structure["countOfDocs"]

            # new_vector = ((pickle.loads(template_structure["binaryTemplateVector"]) * pulled_template[
            #     "countOfDocs"]) + self.feature_array) / pulled_template["countOfDocs"] + 1

            current_dist = np.linalg.norm(updated_vector - TemplateBasedActiveLearning.convert_from_binary(
                image_structure["binaryTemplateVector"]))
            previous_dist = np.linalg.norm(updated_vector - TemplateBasedActiveLearning.convert_from_binary(
                template_structure["closestBinaryTemplateVector"]))

            if current_dist < previous_dist:
                template_structure["closestBinaryTemplateVector"] = image_structure["binaryTemplateVector"]
                blur_image = TemplateBasedActiveLearning.convert_to_blur_image(image)
                cv2.imwrite(Loader.TEMPLATE_FOLDER + template_structure["templateId"] + ".jpg", blur_image)
                template_structure["closestImagePath"] = Loader.TEMPLATE_FOLDER + template_structure[
                    "templateId"] + ".jpg"
            # template_structure["toBeReviewed"].remove(image_structure["fileId"])

        return template_structure, image_structure

    def run(self):

        client = Loader.connect_to_db()
        feature_array = TemplateBasedActiveLearning.get_template_scores(self.image)
        pre_existing_templates = TemplateBasedActiveLearning.get_all_templates(client=client)
        all_possible_templates = TemplateBasedActiveLearning.get_closest_templates(template_list=pre_existing_templates,
                                                                                   target_featurearray=feature_array)

        image_structure = {}
        image_structure["fileId"] = self.file_id
        image_structure["organisation"] = self.organisation
        image_structure["binaryTemplateVector"] = TemplateBasedActiveLearning.convert_to_binary(feature_array)
        resized_img = TemplateBasedActiveLearning.resize_image(self.image)
        image_structure["resizedImage"] = TemplateBasedActiveLearning.convert_to_binary(resized_img)
        print(all_possible_templates)

        if len(all_possible_templates) == 0:
            template_structure = TemplateBasedActiveLearning.create_new_template_structure(file_id=self.file_id,
                                                                                           image=self.image,
                                                                                           organisation=self.organisation,
                                                                                           feature_array=feature_array)

            # image_structure["status"]="algorithmGenerated"
            # image_structure["templateId"]=template_structure["templateId"]
            template_structure, image_structure = TemplateBasedActiveLearning.map_template_to_obj(
                template_structure=template_structure,
                image_structure=image_structure,
                status="algorithmGenerated",
                distance_to_image=0,
                image=self.image)

            image_structure["possibleTemplateIds"] = [template_structure["templateId"]]

            TemplateBasedActiveLearning.insert_record_to_DB(client=client,
                                                            collection_name=Loader.TEMPLATE_COLLECTION_NAME,
                                                            record=template_structure)
            TemplateBasedActiveLearning.insert_record_to_DB(client=client, collection_name=Loader.IMAGE_COLLECTION_NAME,
                                                            record=image_structure)

        else:
            closest_template = TemplateBasedActiveLearning.pull_template_from_DB(
                template_id=all_possible_templates[0]["id"], client=client)
            closest_template, image_structure = TemplateBasedActiveLearning.map_template_to_obj(
                template_structure=closest_template,
                image_structure=image_structure,
                status="algorithmPredicted",
                distance_to_image=all_possible_templates[0]["distance"],
                image=self.image)
            image_structure["possibleTemplateIds"] = [t["id"] for t in all_possible_templates]
            TemplateBasedActiveLearning.update_template_record_to_DB(client=client,
                                                                     collection_name=Loader.TEMPLATE_COLLECTION_NAME,
                                                                     record=closest_template)
            TemplateBasedActiveLearning.update_image_record_to_DB(client=client,
                                                                  collection_name=Loader.IMAGE_COLLECTION_NAME,
                                                                  record=image_structure)

    @staticmethod
    def update(file_id):

        # user_feedback={"template":
        #                     {"status" :"FAIL",
        #                      "attributes":
        #                             {
        #                             "logo":"NA",
        #                             "structure":"NA",
        #                             "colors":"NA",
        #                             "data":"NA",
        #                             },
        #                       "newTemplateId":None
        #
        #                     },
        #                "ruleJson":
        #                    {
        #                        "rule":
        #                            {
        #                                 "newRule":True,
        #                                 "deleteRule":True,
        #                                 "updateRule":True
        #                            },
        #                        "newNormalisation": True,
        #                        "newJson":None
        #
        #                    }
        #             }

        client = Loader.connect_to_db()

        file_doc = TemplateBasedActiveLearning.pull_document_from_DB(file_id=file_id, client=client)
        file_doc_prestat = file_doc["status"]
        file_doc["status"] = "LEARNING"
        TemplateBasedActiveLearning.update_doc_record_to_DB(client=client,
                                                            collection_name=Loader.DOCUMENT_COLLECTION_NAME,
                                                            record=file_doc)

        if not file_doc["pass"]:

            # if user_feedback["template"]["status"] == "FAIL":
            image_structure = TemplateBasedActiveLearning.pull_image_from_DB(file_id=file_id, client=client)
            img = TemplateBasedActiveLearning.convert_from_binary(image_structure["resizedImage"])

            if "selected" not in file_doc["templateFeedback"]:
                # if user_feedback["template"]["newTemplateId"] == "None":

                current_template = TemplateBasedActiveLearning.pull_template_from_DB(image_structure["templateId"],
                                                                                     client=client)
                print("toberev", current_template["toBeReviewed"])
                current_template["toBeReviewed"].remove(image_structure["fileId"])  # check y
                image_to_review = current_template["toBeReviewed"]
                # img = TemplateBasedActiveLearning.convert_from_binary(image_structure["resizedImage"])
                feature_array = TemplateBasedActiveLearning.convert_from_binary(image_structure["binaryTemplateVector"])
                if not file_doc["ruleModification"]:
                    # if user_feedback["ruleJson"]["newJson"] == "None":
                    template_structure = TemplateBasedActiveLearning.create_new_template_structure(
                        file_id=image_structure["fileId"],
                        image=img,
                        organisation=image_structure["organisation"],
                        feature_array=feature_array,
                        t_id=file_doc["templateId"])

                else:
                    with open(Loader.FILES_FOLDER + "/" + file_id + "/" + "hints.json") as f:
                        rule = json.load(f)
                    template_structure = TemplateBasedActiveLearning.create_new_template_structure(
                        file_id=image_structure["fileId"],
                        image=img,
                        organisation=image_structure["organisation"],
                        feature_array=feature_array,
                        rule_json=rule,
                        t_id=file_doc["templateId"]
                    )
                template_structure, image_structure = TemplateBasedActiveLearning.map_template_to_obj(
                    template_structure=template_structure,
                    image_structure=image_structure,
                    status="userGenerated",
                    distance_to_image=0,
                    image=img
                )
                image_structure["possibleTemplateIds"] = [template_structure["templateId"]]
                TemplateBasedActiveLearning.insert_record_to_DB(client=client,
                                                                collection_name=Loader.TEMPLATE_COLLECTION_NAME,
                                                                record=template_structure)
                TemplateBasedActiveLearning.update_image_record_to_DB(client=client,
                                                                      collection_name=Loader.IMAGE_COLLECTION_NAME,
                                                                      record=image_structure)

                for files in image_to_review:
                    print(files)

                    f_doc = TemplateBasedActiveLearning.pull_document_from_DB(file_id=files, client=client)
                    f_doc_prestat = f_doc["status"]
                    f_doc["status"] = "LEARNING"
                    TemplateBasedActiveLearning.update_doc_record_to_DB(client=client,
                                                                        collection_name=Loader.DOCUMENT_COLLECTION_NAME,
                                                                        record=f_doc)

                    file_img = TemplateBasedActiveLearning.convert_from_binary(image_structure["resizedImage"])
                    file_image_structure = TemplateBasedActiveLearning.pull_image_from_DB(file_id=files, client=client)
                    file_feature_array = TemplateBasedActiveLearning.convert_from_binary(
                        file_image_structure["binaryTemplateVector"])
                    pre_existing_templates = TemplateBasedActiveLearning.get_all_templates(client=client)
                    all_possible_templates = TemplateBasedActiveLearning.get_closest_templates(
                        template_list=pre_existing_templates, target_featurearray=file_feature_array)

                    file_closest_template = TemplateBasedActiveLearning.pull_template_from_DB(
                        template_id=all_possible_templates[0]["id"], client=client)
                    file_closest_template, file_image_structure = TemplateBasedActiveLearning.map_template_to_obj(
                        template_structure=file_closest_template,
                        image_structure=file_image_structure,
                        status="algorithmPredicted",
                        distance_to_image=all_possible_templates[0]["distance"],
                        image=file_img)
                    if file_closest_template["templateId"] != current_template["templateId"]:
                        current_template["toBeReviewed"].remove(file_image_structure["fileId"])
                        # file_closest_template["toBeReviewed"].append(file_image_structure["fileId"])

                    file_image_structure["possibleTemplateIds"] = [t["id"] for t in all_possible_templates]

                    TemplateBasedActiveLearning.update_template_record_to_DB(client=client,
                                                                             collection_name=Loader.TEMPLATE_COLLECTION_NAME,
                                                                             record=file_closest_template)
                    TemplateBasedActiveLearning.update_image_record_to_DB(client=client,
                                                                          collection_name=Loader.IMAGE_COLLECTION_NAME,
                                                                          record=file_image_structure)

                    f_doc = TemplateBasedActiveLearning.pull_document_from_DB(file_id=files, client=client)
                    f_doc["status"] = "FETCH"
                    TemplateBasedActiveLearning.update_doc_record_to_DB(client=client,
                                                                        collection_name=Loader.DOCUMENT_COLLECTION_NAME,
                                                                        record=f_doc)

                TemplateBasedActiveLearning.update_template_record_to_DB(client=client,
                                                                         collection_name=Loader.TEMPLATE_COLLECTION_NAME,
                                                                         record=current_template)


            else:

                current_template = TemplateBasedActiveLearning.pull_template_from_DB(image_structure["templateId"],
                                                                                     client=client)
                current_template["toBeReviewed"].remove(image_structure["fileId"])
                TemplateBasedActiveLearning.update_template_record_to_DB(client=client,
                                                                         collection_name=Loader.TEMPLATE_COLLECTION_NAME,
                                                                         record=current_template)
                if file_doc["templateFeedback"]["selected"]["type"] == "OPTIONS":
                    no = file_doc["templateId"]

                user_opted_template = TemplateBasedActiveLearning.pull_template_from_DB(
                    # template_id=user_feedback["template"]["newTemplateId"],
                    template_id=no,
                    client=client)

                user_opted_template, image_structure = TemplateBasedActiveLearning.map_template_to_obj(
                    template_structure=user_opted_template,
                    image_structure=image_structure,
                    status="userRecalibrated",
                    distance_to_image=0,
                    image=img,
                    # feedback=user_feedback)
                    feedback=file_doc)
                TemplateBasedActiveLearning.update_template_record_to_DB(client=client,
                                                                         collection_name=Loader.TEMPLATE_COLLECTION_NAME,
                                                                         record=user_opted_template)
                TemplateBasedActiveLearning.update_image_record_to_DB(client=client,
                                                                      collection_name=Loader.IMAGE_COLLECTION_NAME,
                                                                      record=image_structure)




        elif file_doc["pass"]:

            # elif user_feedback["template"]["status"] == "PASS":
            image_structure = TemplateBasedActiveLearning.pull_image_from_DB(file_id=file_id, client=client)
            template_structure = TemplateBasedActiveLearning.pull_template_from_DB(
                template_id=image_structure["templateId"], client=client)

            image = TemplateBasedActiveLearning.convert_from_binary(image_structure["resizedImage"])
            template_structure, image_structure = TemplateBasedActiveLearning.map_template_to_obj(
                template_structure=template_structure,
                image_structure=image_structure,
                status="userReviewed",
                distance_to_image=0,
                image=image,
                # feedback=user_feedback)
                feedback=file_doc)
            TemplateBasedActiveLearning.update_template_record_to_DB(client=client,
                                                                     collection_name=Loader.TEMPLATE_COLLECTION_NAME,
                                                                     record=template_structure)
            TemplateBasedActiveLearning.update_image_record_to_DB(client=client,
                                                                  collection_name=Loader.IMAGE_COLLECTION_NAME,
                                                                  record=image_structure)

        file_doc = TemplateBasedActiveLearning.pull_document_from_DB(file_id=file_id, client=client)
        file_doc["status"] = "FETCH"
        TemplateBasedActiveLearning.update_doc_record_to_DB(client=client,
                                                            collection_name=Loader.DOCUMENT_COLLECTION_NAME,
                                                            record=file_doc)


if __name__ == "__main__":
    input_list = sys.argv

    if input_list[1] == "run":
        # run
        data_path = input_list[2]
        file_id = (data_path.split("/")[-2])
        organisation = input_list[3]
        if data_path.endswith(".pdf") or data_path.endswith(".PDF"):
            pdf = pdfplumber.open(data_path)
            image = convert_from_path(data_path, first_page=1, last_page=1)[0]
            prep_image = np.asarray(image)
        else:
            prep_image = cv2.imread(data_path)
        print(type(prep_image))
        TemplateBasedActiveLearning(image=prep_image, file_id=str(file_id), organisation=str(organisation))

    elif input_list[1] == "update":
        file_id = input_list[2]
        TemplateBasedActiveLearning.update(file_id=file_id)

        # update

#     # 1 => path to pdf
#     # 2 => "run"
#
#
#     # 1 => file_id
#     # 2 => update
#
#
#
#     # template_folder
#     # input_list=["something.py","/Users/kevin/Downloads/Queries for kevin/ami_1234//RW00275473_2017-12-01.pdf"]
#
#     data_path = input_list[1]
#     image_path = data_path
#     organisation, file_id = (data_path.split("/")[-2]).split("_")
#
#     if data_path.endswith(".pdf") or data_path.endswith(".PDF"):
#         pdf = pdfplumber.open(data_path)
#         image = convert_from_path(data_path, first_page=1, last_page=1)[0]
#         prep_image = np.asarray(image)
#     else:
#         prep_image = cv2.imread(image_path)
#     # cv2.imshow("ip",prep_image)
#     # cv2.waitKey(0)
#     # exit()
#     ActiveLearning(image=prep_image, file_id=str(file_id), organisation=str(organisation))

# if __name__ == "__main__":
#     # path="/Users/kevin/Downloads/template/"
#     # a = [['att_1.jpg', "att", "1"],
#     #     ['REMIT_02406787_page_0.jpg', "remit", "3"],
#     #      ['att_1.jpg', "att", "2"],
#     #      ['AAM_SpireEnergy_300000000192_1804.jpg', "spire", "4"],
#     #      ['att2.jpg', "att", "5"],
#     #      ['att3.jpg', "att", "6"],
#     #      ['att4.jpg', "att", "7"],
#     #      ['att5.jpg', "att", "9"],
#     #      ['att7.jpg', "att", "10"],
#     #      ['att6.jpg', "att", "11"],
#     #      ['AAM_SpireEnergy_7300000000192_1805.jpg', "spire", "12"],
#     #      ['we.jpg', "we", "13"]]
#     # for eachh in a:
#     #     im = cv2.imread(path +eachh[0])
#     #     TemplateBasedActiveLearning(image=im, organisation=eachh[1], file_id=eachh[2])
#     user_dict={}
#     user_feedback = {"template":
#                                          {"status" :"FAIL",
#                                           "attributes":
#                                                  {
#                                                  "logo":"NA",
#                                                  "structure":"NA",
#                                                  "colors":"NA",
#                                                  "data":"NA",
#                                                  },
#                                            "newTemplateId":"c9fd9494-b476-4551-9d51-420f8b01a108"
#                                             # "newTemplateId":"None"
#
#                                          },
#                     "ruleJson":
#                                         {
#                                             "rule":
#                                                 {
#                                                      "newRule":True,
#                                                      "deleteRule":True,
#                                                      "updateRule":True
#                                                 },
#                                             "newNormalisation": True,
#                                             "newJson": {"kevin":"xavier"}
#
#                                         }
#                                  }
#     TemplateBasedActiveLearning.update("4",user_feedback=user_feedback)
# #

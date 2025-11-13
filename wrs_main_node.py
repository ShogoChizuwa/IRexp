#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import json
import os
import traceback
import rospy
import rospkg
import tf2_ros
from std_msgs.msg import String
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper


class WrsMainController():
    """
    WRSのシミュレーション環境内でタスクを実行するクラス
    """
    IGNORE_LIST = ["small_marker", "large_marker", "tuna_fish_can"]
    GRASP_TF_NAME = "object_grasping"
    GRASP_BACK_SAFE = {"z": 0.05, "xy": 0.3}
    GRASP_BACK = {"z": 0.05, "xy": 0.1}
    HAND_PALM_OFFSET = 0.05  # hand_palm_linkは指の付け根なので、把持のために少しずらす必要がある
    HAND_PALM_Z_OFFSET = 0.075
    DETECT_CNT = 5
    TROFAST_Y_OFFSET = 0.2
    SAFETY_BUFFER = 0.42 # 40cm
    GRASPABLE_Y_THRESHOLD = 2.0

    # ルールブック(4.3節)の分類に従い、カテゴリにマッピング
    # [cite: 453-472, 474-485, 487-491, 497, 504-515, 517-524, 481-482, 487-488]
    LABEL_TO_CATEGORY = {
        # 方位に基づくアイテム (Orientation) - 最高得点
        "fork": "orientation",           # [cite: 481]
        "spoon": "orientation",          # [cite: 482]
        "large_marker": "orientation",     # [cite: 487]
        "small_marker": "orientation",     # [cite: 488]

        # 食品 (Food) [cite: 453-472]
        "cracker_box": "food",
        "sugar_box": "food",
        "pudding_box": "food",
        "gelatin_box": "food",
        "potted_meat_can": "food",
        "master_chef_can": "food",
        "tuna_fish_can": "food",
        "chips_can": "food",
        "mustard_bottle": "food",
        "tomato_soup_can": "food",
        "banana": "food",
        "strawberry": "food",
        "apple": "food",
        "lemon": "food",
        "peach": "food",
        "pear": "food",
        "orange": "food",
        "plum": "food",

        # キッチン用品 (Kitchen) [cite: 474-485]
        "windex_bottle": "kitchen",
        "bleach_cleanser": "kitchen",
        "sponge": "kitchen",
        "pitcher_base": "kitchen",
        "pitcher_lid": "kitchen",
        "plate": "kitchen",
        "bowl": "kitchen",
        "spatula": "kitchen",
        "wine_glass": "kitchen",
        "mug": "kitchen",

        # ツール (Tools) [cite: 489-491]
        "padlock": "tools", # 'padlock' (鍵 [cite: 489])
        "bolt_and_nut": "tools",
        "clamp": "tools",

        # 形状アイテム (Shape) [cite: 497, 504-515]
        "credit_card_blank": "shape",
        "mini_soccer_ball": "shape",
        "softball": "shape",
        "baseball": "shape",
        "tennis_ball": "shape",
        "racquetball": "shape",
        "golf_ball": "shape",
        "marble": "shape",
        "cup": "shape",
        "foam_brick": "shape",
        "dice": "shape",
        "rope": "shape",
        "chain": "shape",

        # タスク項目 (Task) [cite: 517-518]
        "rubiks_cube": "task",
        "colored_wood_block": "task"
    }

    # ルールブック(表2)に基づき、カテゴリを配置場所(座標名)にマッピング
    CATEGORY_TO_PLACE = {
        "shape": "drawer_left_place",      # 形状アイテム -> 左引き出し 
        "tools": "drawer_top_place",       # ツール -> 引き出し上部  (または "drawer_bottom_place")
        "food": "tray_a_place",          # 食品 -> トレイA  (または "tray_b_place")
        "kitchen": "container_a_place",    # キッチン用品 -> 容器 A 
        "orientation": "container_b_place",# 方位に基づくアイテム -> コンテナB 
        "task": "bin_a_place",           # タスク項目 -> ビンA 
        "unknown": "bin_b_place"         # 未知の物体 -> ビンB 
    }

    def __init__(self):
        # 変数の初期化
        self.instruction_list = []
        self.detection_list = []

        self.food_counter = 0  # 食品カテゴリを置いた回数をカウント

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses = self.load_json(self.get_path(["config", "poses.json"]))

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.instruction_sub = rospy.Subscriber("/message",    String, self.instruction_cb,
                                                queue_size=10)
        self.detection_sub   = rospy.Subscriber("/detect_msg", String, self.detection_cb,
                                                queue_size=10)

    @staticmethod
    def get_path(pathes, package="wrs_algorithm"):
        """
        ROSパッケージ名とファイルまでのパスを指定して、ファイルのパスを取得する
        """
        if not pathes:  # check if the list is empty
            rospy.logerr("Can NOT resolve file path.")
            raise ValueError("You must specify the path to file.")
        pkg_path = rospkg.RosPack().get_path(package)
        path = os.path.join(*pathes)
        return os.path.join(pkg_path, path)

    @staticmethod
    def load_json(path):
        """
        jsonファイルを辞書型で読み込む
        """
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)

    def instruction_cb(self, msg):
        """
        指示文を受信する
        """
        rospy.loginfo("instruction received. [%s]", msg.data)
        self.instruction_list.append(msg.data)

    def detection_cb(self, msg):
        """
        検出結果を受信する
        """
        rospy.loginfo("received [Collision detected with %s]", msg.data)
        self.detection_list.append(msg.data)

    def get_relative_coordinate(self, parent, child):
        """
        tfで相対座標を取得する
        """
        try:
            # 4秒待機して各tfが存在すれば相対関係をセット
            trans = self.tf_buffer.lookup_transform(parent, child,rospy.Time.now(),
                                                    rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = f"failed to get transform between [{parent}] and [{child}]\n"
            log_str += traceback.format_exc()
            rospy.logerr(log_str)
            return None

    def goto_name(self, name):
        """
        waypoint名で指定された場所に移動する
        """
        if name in self.coordinates["positions"].keys():
            pos = self.coordinates["positions"][name]
            rospy.loginfo("go to [%s](%.2f, %.2f, %.2f)", name, pos[0], pos[1], pos[2])
            return omni_base.go_abs(pos[0], pos[1], pos[2])

        rospy.logerr("unknown waypoint name [%s]", name)
        return False

    def goto_pos(self, pos):
        """
        waypoint名で指定された場所に移動する
        """
        rospy.loginfo("go to [raw_pos](%.2f, %.2f, %.2f)", pos[0], pos[1], pos[2])
        return omni_base.go_abs(pos[0], pos[1], pos[2])

    def change_pose(self, name):
        """
        指定された姿勢名に遷移する
        """
        if name in self.poses.keys():
            rospy.loginfo("change pose to [%s]", name)
            return whole_body.move_to_joint_positions(self.poses[name])

        rospy.logerr("unknown pose name [%s]", name)
        return False

    def check_positions(self):
        """
        読み込んだ座標ファイルの座標を巡回する
        """
        whole_body.move_to_go()
        for wp_name in self.coordinates["routes"]["test"]:
            self.goto_name(wp_name)
            rospy.sleep(1)

    def get_latest_detection(self):
        """
        最新の認識結果が到着するまで待つ
        """
        res = self.detection_clt(GetObjectDetectionRequest())
        return res.bboxes

    def get_grasp_coordinate(self, bbox):
        """
        BBox情報から把持座標を取得する
        """
        # BBox情報からtfを生成して、座標を取得
        self.tf_from_bbox_clt.call(SetTransformFromBBoxRequest(bbox=bbox, frame=self.GRASP_TF_NAME))
        rospy.sleep(1.0)  # tfが安定するのを待つ
        return self.get_relative_coordinate("map", self.GRASP_TF_NAME).translation

    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None:
            return None
        return obj["bbox"]

    @classmethod
    def get_most_graspable_obj(cls, obj_list):
        """
        把持すべきscoreが最も高い物体を返す。
        """
        extracted = []
        extract_str = "detected object list\n"
        ignore_str  = ""
        for obj in obj_list:
            info_str = (
                f"{obj.label:<15}({obj.score:.2%}, "
                f"{obj.x:3d}, {obj.y:3d}, {obj.w:3d}, {obj.h:3d})\n")
            if obj.label in cls.IGNORE_LIST:
                ignore_str += "- ignored  : " + info_str
            else:
                score = cls.calc_score_bbox(obj)
                extracted.append({"bbox": obj, "score": score, "label": obj.label})
                extract_str += f"- extracted: {score:07.3f} " + info_str

        rospy.loginfo(extract_str + ignore_str)

        # つかむべきかのscoreが一番高い物体を返す
        for obj_info in sorted(extracted, key=lambda x: x["score"], reverse=True):
            obj      = obj_info["bbox"]
            info_str = (
                f"{obj.label} ({obj.score:.2%}, {obj.x:3d}, {obj.y:3d}, {obj.w:3d}, {obj.h:3d})\n")
            rospy.loginfo("selected bbox: " + info_str)
            return obj_info

        # objが一つもない場合は、Noneを返す
        return None

    @classmethod
    def calc_score_bbox(cls, bbox):
        """
        detector_msgs/BBoxのスコアを計算する
        """
        gravity_x = bbox.x + bbox.w / 2
        gravity_y = bbox.y + bbox.h / 2
        xy_diff   = abs(320- gravity_x) / 320 + abs(360 - gravity_y) / 240

        return 1 / xy_diff

    @classmethod
    def get_most_graspable_bboxes_by_label(cls, obj_list, label):
        """
        label名が一致するオブジェクトの中から最も把持すべき物体のbboxを返す
        """
        match_objs = [obj for obj in obj_list if obj.label in label]
        if not match_objs:
            rospy.logwarn("Cannot find a object which labeled with similar name.")
            return None
        # 2. ラベルが一致したリスト(match_objs)を、
        #    検出確率(obj.score)が高い順にソートする
        sorted_by_score = sorted(match_objs, key=lambda obj: obj.score, reverse=True)
        # 3. 最も確率の高かった物体（リストの先頭）を返す
        best_match = sorted_by_score[0]
        info_str = (
            f"{best_match.label} ({best_match.score:.2%}, {best_match.x:3d}, "
            f"{best_match.y:3d}, {best_match.w:3d}, {best_match.h:3d})\n")

        rospy.loginfo("Selected best match by probability: " + info_str)

        return best_match

    @staticmethod
    def extract_target_obj_and_person(instruction):
        """
        指示文("物体名 to 人" 形式)から物体と人物を抽出する
        """
        rospy.loginfo("[extract_target_obj_and_person] instruction:"+ instruction)
        # デフォルト値（解析失敗時に使用）
        target_obj = "apple"
        target_person = "right"
        # --- ここから解析ロジック ---
        # " to person " という文字列で指示を分割する
        separator = " to person "
        if separator in instruction:
            parts = instruction.split(separator)
            if len(parts) == 2:
                # 正常に "物体名" と "人物名" に分割できた場合
                target_obj = parts[0].strip() # " to person " の前が物体名
                target_person = parts[1].strip() # " to person " の後が人物名
            # (例: "colored_wood_block to person right" -> "colored_wood_block" と "right")
                rospy.loginfo(f" -> Parsed object: [{target_obj}], person: [{target_person}]")
            else:
                # 形式が予期せず異なる場合 (例: "A to person B to person C")
                rospy.logerr(" -> Failed to parse instruction (unexpected format). Using defaults.")
        else:
            # 指示に " to person " が含まれていない場合
            rospy.logerr(
                f" -> Failed to parse instruction (separator '{separator}' not found)."
                f" Using defaults.")
        # --- 解析ここまで ---

        return target_obj, target_person
    def get_placement_info(self, label):
        """
        検出した物体のラベル(label)から、
        それが属するカテゴリ(category)と、
        置くべき場所(place)の座標名を返す。
        """
        # 1. ラベルからカテゴリを特定
        # 辞書にないラベルの場合は 'unknown' カテゴリとする
        category = self.LABEL_TO_CATEGORY.get(label, "unknown")

        # 2. カテゴリから配置場所を特定
        # カテゴリが'unknown'の場合、CATEGORY_TO_PLACE['unknown'] (bin_b_place) が返される
        place = self.CATEGORY_TO_PLACE.get(category, "bin_b_place")

        if category == "food":
            if self.food_counter % 2 == 0:
                place = "tray_a_place"
            else:
                place = "tray_b_place"

        self.food_counter += 1 # カウンターを増やす

        # 3. カテゴリと場所を返す
        rospy.loginfo(f"Label: '{label}' -> Category: '{category}' -> Place: '{place}'")
        return category, place

    def grasp_from_side(self, pos_x, pos_y, pos_z, yaw, pitch, roll, preliminary="-y"):
        """
        把持の一連の動作を行う

        NOTE: tall_tableに対しての予備動作を生成するときはpreliminary="-y"と設定することになる。
        """
        if preliminary not in [ "+y", "-y", "+x", "-x" ]:
            raise RuntimeError(f"unnkown graps preliminary type [{preliminary}]")

        rospy.loginfo("move hand to grasp (%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        grasp_back_safe = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_back = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_pos = {"x": pos_x, "y": pos_y, "z": pos_z}

        sign = 0
        if "+" in preliminary:
            sign = 1
        elif "-" in preliminary:
            sign = -1

        if "x" in preliminary:
            grasp_back_safe["x"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["x"] += sign * self.GRASP_BACK["xy"]
        elif "y" in preliminary:
            grasp_back_safe["y"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["y"] += sign * self.GRASP_BACK["xy"]

        gripper.command(1)
        whole_body.move_end_effector_pose(
            grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            grasp_back["x"], grasp_back["y"], grasp_back["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            grasp_pos["x"], grasp_pos["y"], grasp_pos["z"], yaw, pitch, roll)
        gripper.command(0)
        # グリッパーが完全に閉じるのを待つ
        rospy.sleep(1.0) # (例: 1.0秒待機．掴む物体に応じて調整)
        whole_body.move_end_effector_pose(
            grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)

    def grasp_from_front_side(self, grasp_pos, _bbox=None):
        """62312798 千頭和翔梧
        正面把持を行う
        ややアームを下に向けている
        """
        grasp_pos.y -= self.HAND_PALM_OFFSET
        rospy.loginfo("grasp_from_front_side (%.2f, %.2f, %.2f)",
                      grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -100, 0, "-y")

    def grasp_from_upper_side(self, grasp_pos, bbox=None):
        """62312798 千頭和翔梧
        上面から把持を行う
        オブジェクトに寄るときは、y軸から近づく上面からは近づかない
        """
        if bbox.h > 2 * self.HAND_PALM_Z_OFFSET:
            grasp_pos.z += self.HAND_PALM_Z_OFFSET
        rospy.loginfo("grasp_from_upper_side (%.2f, %.2f, %.2f)",
                      grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -160, 0, "-y")

    def exec_graspable_method(self, grasp_pos, label="", bbox=None):
        """62312798 千頭和翔梧
        task1専用:posの位置によって把持方法を判定し実行する。
        """
        method = None
        graspable_y = 1.85  # これ以上奥は把持できない
        desk_y = 1.5
        desk_z = 0.35

        # 把持禁止判定
        if graspable_y < grasp_pos.y and desk_z > grasp_pos.z:
            return False

        if label in ["cup", "frisbee", "bowl"]:
            # bowlの張り付き対策
            method = self.grasp_from_upper_side
        else:
            if desk_y < grasp_pos.y and desk_z > grasp_pos.z:
                # 机の下である場合
                method = self.grasp_from_front_side
            else:
                method = self.grasp_from_upper_side

        method(grasp_pos, bbox)
        return True

    def put_in_place(self, place, into_pose):
        """
        指定場所に入れ、all_neutral姿勢を取る。
        """
        self.change_pose("look_at_near_floor")
        self.goto_name(place)
        self.change_pose("all_neutral")
        self.change_pose(into_pose)
        gripper.command(1)
        rospy.sleep(5.0)
        self.change_pose("all_neutral")

    def pull_out_trofast(self, x, y, z, yaw, pitch, roll, left = False):
        """62312798 千頭和翔梧
        trofastの引き出しを引き出す
        """
        if left:
            self.goto_name("stair_like_drawer2")
        else:
            self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        gripper.command(1)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(x, y, z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw,  pitch, roll)
        gripper.command(1)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET * 1.5, z, yaw,  pitch, roll)
        self.change_pose("all_neutral")

    def push_in_trofast(self, pos_x, pos_y, pos_z, yaw, pitch, roll):
        """
        trofastの引き出しを戻す
        NOTE:サンプル
            self.push_in_trofast(0.178, -0.29, 0.75, -90, 100, 0)
        """
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        pos_y+=self.HAND_PALM_OFFSET

        # 予備動作-押し込む
        whole_body.move_end_effector_pose(pos_x, pos_y + self.TROFAST_Y_OFFSET * 1.5, pos_z,
                                          yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(pos_x, pos_y + self.TROFAST_Y_OFFSET, pos_z,
                                          yaw, pitch, roll)
        whole_body.move_end_effector_pose(pos_x, pos_y, pos_z, yaw, pitch, roll)

        self.change_pose("all_neutral")

    def find_best_grasp_target(self):
        """62312798 千頭和翔梧
        現在の視界から、最も掴みやすい物体（候補）を探す。
        フィルターとスコアリングを実行し、最適な候補（辞書）を返す。
        見つからなければ None を返す。
        """
        # 1. 検出した全物体を取得
        detected_objs_list = self.get_latest_detection()

        # 2. フィルターを通過した「掴める候補」のリスト
        grasp_candidates = []

        # 3. 全ての検出物体をチェック
        for obj in detected_objs_list.bboxes:

            # 4. IGNORE_LIST チェック
            if obj.label in self.IGNORE_LIST:
                rospy.loginfo("Ignoring object [%s] (In IGNORE_LIST)", obj.label)
                continue

            # 5. 物体の3D座標を取得
            grasp_pos = self.get_grasp_coordinate(obj)
            if grasp_pos is None:
                rospy.logwarn("Failed to get coordinate for [%s]", obj.label)
                continue

            # 6. Y座標フィルターを実行
            if grasp_pos.y >= self.GRASPABLE_Y_THRESHOLD:
                rospy.loginfo("Ignoring object [%s] (Too far: Y=%.2f)",
                            obj.label, grasp_pos.y)
                continue

            # 7. 全てのフィルターを通過した物体のみ、候補リストに追加
            grasp_candidates.append({
                "bbox": obj, 
                "score": self.calc_score_bbox(obj), # score変数をインライン化
                "label": obj.label, 
                "pos": grasp_pos
            })

        # 8. 掴める候補が一つもなかった場合
        if not grasp_candidates:
            rospy.logwarn("Cannot find graspable object (After ALL filters).")
            return None # 候補なし

        # 9. 候補リストをスコアの高い順にソート
        grasp_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 10. 最もスコアの高い物体を返す
        return grasp_candidates[0]

    def deliver_to_target(self, target_obj, target_person):
        """
        棚で取得したものを人に渡す。
        """
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")
        self.change_pose("look_at_shelf")

        rospy.loginfo("target_obj: " + target_obj + "  target_person: " + target_person)
        # 物体検出結果から、把持するbboxを決定
        detected_objs = self.get_latest_detection()
        grasp_bbox = self.get_most_graspable_bboxes_by_label(detected_objs.bboxes, target_obj)
        if grasp_bbox is None:
            rospy.logwarn("Cannot find object to grasp. task2b is aborted.")
            return
        rospy.loginfo(f"Found graspable object '{target_obj}'")

        # BBoxの3次元座標を取得して、その座標で把持する
        grasp_pos = self.get_grasp_coordinate(grasp_bbox)
        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos)
        # 掴んだ状態を安定させる
        rospy.sleep(0.5) # (例: 0.5秒待機)
        self.change_pose("all_neutral")
        # --- 配達先決定の修正 (person_b 固定 [cite: 569] をやめる) ---
        # 'target_person' (例: "right") に基づいて配達先を決定
        delivery_location = ""
        if target_person == "right":
            delivery_location = "person_b" # "right" を "person_b" (座標名) にマッピング
        elif target_person == "left":
            delivery_location = "person_a" # "left" を "person_a" (座標名) にマッピング
        else:
            # デフォルト（"right"でも"left"でもない場合）
            rospy.logwarn(f"Unknown person name '{target_person}', defaulting to 'person_b'")
            delivery_location = "person_b"

        rospy.loginfo(f"Delivering to: {delivery_location} (based on '{target_person}')")
        # target_personの前に持っていく
        self.change_pose("look_at_near_floor")
        self.goto_name(delivery_location)
        self.change_pose("deliver_to_human")
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")

    def execute_avoid_blocks(self):
        # TODO 自分の名前とは別に、関数の説明をするDocstringを記述してください
        # 完了したらTODO含めこれらのコメントを削除すること

        # [統合安全チェック] X軸移動を先に実行し、安全を確保してからY軸で前進

        # 10ステップのY座標境界
        y_thresholds = [1.85, 1.995, 2.14, 2.285, 2.43, 2.575, 2.72, 2.865, 3.01, 3.155, 3.3]

        for i in range(10): # 10ステップのループ

            # --- 1. 現在地の取得 ---
            try:
                trans = self.tf_buffer.lookup_transform("map", "base_link",
                                                        rospy.Time(0), rospy.Duration(1.0))
                current_x = trans.transform.translation.x
                current_y = trans.transform.translation.y
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logerr("Could not get current robot pose: %s", e)
                continue

            # --- 2. 次の目標Y座標 ---
            target_y = y_thresholds[i+1]

            # --- 3. 障害物検知 ---
            detected_objs = self.get_latest_detection()
            pos_bboxes = [self.get_grasp_coordinate(bbox) for bbox in detected_objs.bboxes]

            # --- 4. 最適なレーンを選択 (横移動+前進の両方が安全な場所) ---
            target_x = self.select_best_safe_lane(pos_bboxes, current_x, current_y, target_y)

            # --- 5. 実行: まず横移動 (X) ---
            if abs(target_x - current_x) > 0.05: # 5cm以上の移動が必要なら
                rospy.loginfo("Step %d: Adjusting X to %.2f (at Y=%.2f)", i, target_x, current_y)
                self.goto_pos([target_x, current_y, 90])
                rospy.sleep(0.5) # X移動の完了を待つ

            # --- 6. 実行: 次に前進 (Y) ---
            rospy.loginfo("Step %d: Advancing Y to %.2f (in lane X=%.2f)", i, target_y, target_x)
            self.goto_pos([target_x, target_y + 0.1, 90])
            rospy.sleep(0.5) # Y移動の完了を待つ

        rospy.loginfo("Finished integrated avoid blocks.")

    def select_best_safe_lane(self, pos_bboxes, current_x, current_y, target_y):
        # TODO ローカル変数が多すぎるので減らしてください。
        """
        [NEW - 統合安全チェック]
        現在の(current_x, current_y)から、あるレーンへ横移動し、
        そこからtarget_yまで前進する経路がすべて安全かどうかを判定する。
        """
        # レーン定義
        interval = 0.45
        pos_xa = 1.7
        pos_xb = 1.8 + interval       # 2.25
        pos_xc = 1.7 + (interval * 2)  # 2.60
        lane_centers = {"xa": pos_xa, "xb": pos_xb, "xc": pos_xc}

        # 安全マージン (ロボット半径 + 障害物半径 + バッファ)
        # self.SAFETY_BUFFER, グローバル変数として設定

        # 各レーンの安全フラグ (Trueで初期化)
        lane_safe = {"xa": True, "xb": True, "xc": True}

        for pos in pos_bboxes:
            if pos is None:
                continue

            # --- チェック1: 横移動の安全性 ---
            # 現在のY座標付近(±20cm)にある障害物について
            if (current_y - 0.2) < pos.y < (current_y + 0.2):
                for lane_name, lane_x in lane_centers.items():
                    # 現在地からこのレーンへの横移動パス(X)と障害物(pos.x)が重なるか？
                    path_min_x = min(current_x, lane_x) - self.SAFETY_BUFFER
                    path_max_x = max(current_x, lane_x) + self.SAFETY_BUFFER
                    if path_min_x < pos.x < path_max_x:
                        lane_safe[lane_name] = False  # このレーンへの横移動は危険
                        rospy.logwarn("X-Path to %s blocked by obstacle at (%.2f, %.2f)",
                                      lane_name, pos.x, pos.y)

            # --- チェック2: 前進の安全性 ---
            # これから進むY座標(current_y ~ target_y)の間にある障害物について
            if current_y < pos.y < target_y:
                for lane_name, lane_x in lane_centers.items():
                    # このレーンの前進パス(X)と障害物(pos.x)が重なるか？
                    if (lane_x - self.SAFETY_BUFFER) < pos.x < (lane_x + self.SAFETY_BUFFER):
                        lane_safe[lane_name] = False  # このレーンでの前進は危険
                        rospy.logwarn("Y-Path in %s blocked by obstacle at (%.2f, %.2f)",
                                      lane_name, pos.x, pos.y)

        # 安全なレーン候補を抽出
        safe_candidates = [lane for lane, is_safe in lane_safe.items() if is_safe]

        rospy.loginfo("Safe lanes for Step: %s", safe_candidates)

        if not safe_candidates:
            rospy.logwarn("NO COMPLETELY SAFE LANE! " +
                          "Staying in current X lane as emergency fallback.")
            return current_x  # 危険なら動かないのが一番マシ

        # 安全な候補の中で、現在地に最も近いレーンを選ぶ（無駄な動きを減らす）
        best_lane = min(safe_candidates, key=lambda l: abs(lane_centers[l] - current_x))
        rospy.loginfo("Selected best (closest safe) lane: %s", best_lane)
        return lane_centers[best_lane]

    def execute_task1(self):
        """62312798 千頭和翔梧
        task1を実行する
        """
        rospy.loginfo("#### start Task 1 ####")
        hsr_position = [
            ("near_long_table_l", "look_at_near_floor"),
            ("near_tall_table", "look_at_near_floor"),
            ("tall_table", "look_at_tall_table"),
            ("long_table_r", "look_at_tall_table"),
        ]

        self.pull_out_trofast(0.135, -0.3, 0.275, 0, -100, 90)
        self.pull_out_trofast(0.135, -0.31, 0.545, 0, -100, 90)
        self.pull_out_trofast(0.48, -0.31, 0.275, 0, -100, 90, True)

        for plc, pose in hsr_position:
            # 正面経由
            front_waypoint_name = plc + "_front"
            rospy.loginfo("Going to front of %s (via %s)", plc, front_waypoint_name)
            self.goto_name(front_waypoint_name)

            for _ in range(self.DETECT_CNT):
                # 移動と視線指示
                self.goto_name(plc)
                self.change_pose(pose)
                gripper.command(0)

                # 1. 検出ロジックをヘルパー関数に切り出し (ローカル変数を大幅削減)
                best_obj_info = self.find_best_grasp_target()

                # 2. 候補が見つからなかった場合はリトライ
                if best_obj_info is None:
                    continue # 次の検出試行へ

                # 3. 変数展開をせず、辞書のまま使う (ローカル変数削減)
                rospy.loginfo("grasp the " + best_obj_info["label"])

                # 4. 把持を実行
                self.change_pose("grasp_on_table")

                if not self.exec_graspable_method(
                    best_obj_info["pos"],
                    best_obj_info["label"],
                    best_obj_info["bbox"]
                ):
                    rospy.logwarn("exec_graspable_method returned False for [%s]",
                                  best_obj_info["label"])
                    continue

                self.change_pose("all_neutral")

                # 5. ラベルからカテゴリと配置場所を取得
                _category, place_name = self.get_placement_info(best_obj_info["label"])

                # 6. 姿勢を決定
                if place_name == "drawer_top_place":
                    into_pose = "put_in_drawer"
                else:
                    into_pose = "put_in_bin"

                # 7. 取得した配置場所(place_name)と、固定の姿勢(into_pose)で物体を置く
                self.put_in_place(place_name, into_pose)

    def execute_task2a(self):
        """
        task2aを実行する
        """
        rospy.loginfo("#### start Task 2a ####")
        self.change_pose("look_at_near_floor")
        gripper.command(0)
        self.change_pose("look_at_near_floor")
        self.goto_name("standby_2a")

        # 落ちているブロックを避けて移動
        self.execute_avoid_blocks()

        self.goto_name("go_throw_2a")
        whole_body.move_to_go()

    def execute_task2b(self):
        """
        task2bを実行する
        """
        rospy.loginfo("#### start Task 2b ####")
        # 命令文を取得
        if self.instruction_list:
            latest_instruction = self.instruction_list[-1]
            rospy.loginfo("recieved instruction: %s", latest_instruction)
        else:
            rospy.logwarn("instruction_list is None")
            return

        # 命令内容を解釈
        target_obj, target_person = self.extract_target_obj_and_person(latest_instruction)

        # 指定したオブジェクトを指定した配達先へ
        if target_obj and target_person:
            self.deliver_to_target(target_obj, target_person)

    def run(self):
        """
        全てのタスクを実行する
        """
        self.change_pose("all_neutral")
        self.execute_task1()
        self.execute_task2a()
        self.execute_task2b()


def main():
    """
    WRS環境内でタスクを実行するためのメインノードを起動する
    """
    rospy.init_node('main_controller')
    try:
        ctrl = WrsMainController()
        rospy.loginfo("node initialized [%s]", rospy.get_name())

        # タスクの実行モードを確認する
        if rospy.get_param("~test_mode", default=False) is True:
            rospy.loginfo("#### start with TEST mode. ####")
            ctrl.check_positions()
        else:
            rospy.loginfo("#### start with NORMAL mode. ####")
            ctrl.run()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

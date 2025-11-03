#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import json
import os
from select import select
import traceback
from turtle import pos
import rospy
import rospkg
import tf2_ros
from std_msgs.msg import String
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper
import math


class WrsMainController(object):
    """
    WRSのシミュレーション環境内でタスクを実行するクラス
    """
    IGNORE_LIST = []
    GRASP_TF_NAME = "object_grasping"
    GRASP_BACK_SAFE = {"z": 0.05, "xy": 0.3}
    GRASP_BACK = {"z": 0.05, "xy": 0.1}
    HAND_PALM_OFFSET = 0.05  # hand_palm_linkは指の付け根なので、把持のために少しずらす必要がある
    HAND_PALM_Z_OFFSET = 0.075
    DETECT_CNT = 5
    TROFAST_Y_OFFSET = 0.2

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
        self.detection_list   = []

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses       = self.load_json(self.get_path(["config", "poses.json"]))

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.instruction_sub = rospy.Subscriber("/message",    String, self.instruction_cb, queue_size=10)
        self.detection_sub   = rospy.Subscriber("/detect_msg", String, self.detection_cb,   queue_size=10)

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
        with open(path, "r") as json_file:
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
            trans = self.tf_buffer.lookup_transform(parent, child,rospy.Time.now(),rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = "failed to get transform between [{}] and [{}]\n".format(parent, child)
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
        self.tf_from_bbox_clt.call(            SetTransformFromBBoxRequest(bbox=bbox, frame=self.GRASP_TF_NAME))
        rospy.sleep(1.0)  # tfが安定するのを待つ
        return self.get_relative_coordinate("map", self.GRASP_TF_NAME).translation

    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None: return None
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
            info_str = "{:<15}({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h)
            if obj.label in cls.IGNORE_LIST:
                ignore_str += "- ignored  : " + info_str
            else:
                score = cls.calc_score_bbox(obj)
                extracted.append({"bbox": obj, "score": score, "label": obj.label})
                extract_str += "- extracted: {:07.3f} ".format(score) + info_str

        rospy.loginfo(extract_str + ignore_str)

        # つかむべきかのscoreが一番高い物体を返す
        for obj_info in sorted(extracted, key=lambda x: x["score"], reverse=True):
            obj      = obj_info["bbox"]
            info_str = "{} ({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h )
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
        return cls.get_most_graspable_bbox(match_objs)

    @staticmethod
    def extract_target_obj_and_person(instruction):
        """
        指示文から対象となる物体名称を抽出する
        """
        #TODO: 関数は未完成です。引数のinstructionを利用すること
        rospy.loginfo("[extract_target_obj_and_person] instruction:"+  instruction)
        target_obj    = "apple"
        target_person = "right"

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
        
        # 3. カテゴリと場所を返す
        rospy.loginfo("Label: '{}' -> Category: '{}' -> Place: '{}'".format(label, category, place))
        return category, place

    def grasp_from_side(self, pos_x, pos_y, pos_z, yaw, pitch, roll, preliminary="-y"):
        """
        把持の一連の動作を行う

        NOTE: tall_tableに対しての予備動作を生成するときはpreliminary="-y"と設定することになる。
        """
        if preliminary not in [ "+y", "-y", "+x", "-x" ]: raise RuntimeError("unnkown graps preliminary type [{}]".format(preliminary))

        rospy.loginfo("move hand to grasp (%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        grasp_back_safe = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_back = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_pos = {"x": pos_x, "y": pos_y, "z": pos_z}

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
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose( grasp_back["x"], grasp_back["y"], grasp_back["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            grasp_pos["x"], grasp_pos["y"], grasp_pos["z"], yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)

    def grasp_from_front_side(self, grasp_pos):
        """
        正面把持を行う
        ややアームを下に向けている
        """
        grasp_pos.y -= self.HAND_PALM_OFFSET
        rospy.loginfo("grasp_from_front_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -100, 0, "-y")

    def grasp_from_upper_side(self, grasp_pos):
        """
        上面から把持を行う
        オブジェクトに寄るときは、y軸から近づく上面からは近づかない
        """
        grasp_pos.z += self.HAND_PALM_Z_OFFSET
        rospy.loginfo("grasp_from_upper_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -160, 0, "-y")

    def exec_graspable_method(self, grasp_pos, label=""):
        """
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

        method(grasp_pos)
        return True

    def put_in_place(self, place, into_pose):
        # 指定場所に入れ、all_neutral姿勢を取る。
        self.change_pose("look_at_near_floor")
        a = "go_palce" # TODO 不要な変数
        self.goto_name(place)
        self.change_pose("all_neutral")
        self.change_pose(into_pose)
        gripper.command(1)
        rospy.sleep(5.0)
        self.change_pose("all_neutral")

    def pull_out_trofast(self, x, y, z, yaw, pitch, roll):
        # trofastの引き出しを引き出す
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        a = True  # TODO 不要な変数
        gripper.command(1)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(x, y, z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw,  pitch, roll)
        gripper.command(1)
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
        whole_body.move_end_effector_pose( pos_x, pos_y +    self.TROFAST_Y_OFFSET * 1.5, pos_z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(  pos_x, pos_y + self.TROFAST_Y_OFFSET, pos_z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(            pos_x, pos_y, pos_z, yaw, pitch, roll)

        self.change_pose("all_neutral")

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

        # BBoxの3次元座標を取得して、その座標で把持する
        grasp_pos = self.get_grasp_coordinate(grasp_bbox)
        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos)
        self.change_pose("all_neutral")

        # target_personの前に持っていく
        self.change_pose("look_at_near_floor")
        self.goto_name("person_b")    # TODO: 配達先が固定されているので修正
        self.change_pose("deliver_to_human")
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")

    def execute_avoid_blocks(self):
        # blockを避ける
        for i in range(10):
            detected_objs = self.get_latest_detection()
            bboxes = detected_objs.bboxes
            pos_bboxes = [self.get_grasp_coordinate(bbox) for bbox in bboxes]
            waypoint = self.select_next_waypoint(i, pos_bboxes)
            # TODO メッセージを確認するためコメントアウトを外す
            # rospy.loginfo(waypoint)
            self.goto_pos(waypoint)

    def select_next_waypoint(self, current_stp, pos_bboxes):
        """
        waypoints から近い場所にあるものを除外し、最適なwaypointを返す。
        x座標を原点に近い方からxa,xb,xcに定義する。bboxを判断基準として移動先を決定する(デフォルトは0.45間隔)
        pos_bboxesは get_grasp_coordinate() 済みであること
        """
        interval = 0.45
        pos_xa = 1.7
        pos_xb = pos_xa + interval
        pos_xc = pos_xb + interval

        # xa配列はcurrent_stpに関係している
        waypoints = {
            "xa": [ [pos_xa, 1.995, 45], [pos_xa, 2.14, 45], [pos_xa, 2.285, 45], [pos_xa, 2.43, 45], [pos_xa, 2.575, 45],
                    [pos_xa, 2.72, 90], [pos_xa, 2.865, 90], [pos_xa, 3.01, 90], [pos_xa, 3.155, 90], [pos_xa, 3.3, 90] ], 
            "xb": [ [pos_xb, 1.995, 90], [pos_xb, 2.14, 90], [pos_xb, 2.285, 90], [pos_xb, 2.43, 90], [pos_xb, 2.575, 90],
                    [pos_xb, 2.72, 90], [pos_xb, 2.865, 90], [pos_xb, 3.01, 90], [pos_xb, 3.155, 90], [pos_xb, 3.3, 90] ],
            "xc": [ [pos_xc, 1.995, 135], [pos_xc, 2.14, 135], [pos_xc, 2.285, 135], [pos_xc, 2.43, 135], [pos_xc, 2.575, 135],
                    [pos_xc, 2.72, 90], [pos_xc, 2.865, 90], [pos_xc, 3.01, 90], [pos_xc, 3.155, 90], [pos_xc, 3.3, 90] ]
        }
        #y座標で場合分け
        y_thresholds = [1.85, 1.995, 2.14, 2.285, 2.43, 2.575, 2.72, 2.865, 3.01, 3.155, 3.3]
        #現在のyと次のy
        current_y = y_thresholds[current_stp]
        next_y = y_thresholds[current_stp + 1]
        # posがxa,xb,xcのラインに近い場合は候補から削除
        is_to_xa = True
        is_to_xb = True
        is_to_xc = True

        for bbox in pos_bboxes:
            pos_x = bbox.x
            pos_y = bbox.y
            # TODO デバッグ時にコメントアウトを外す
            # rospy.loginfo("detected object obj.x = {:.2f}".format(bbox.x))

            # NOTE Hint:ｙ座標次第で無視してよいオブジェクトもある。
            if not (current_y < pos_y < next_y):
                # rospy.loginfo("  -> Ignored (Out of Y-Range)")
                continue  # 判定範囲外の障害物は無視する
            if pos_x < pos_xa + (interval/2):
                is_to_xa = False
                # rospy.loginfo("is_to_xa=False")
                continue
            elif pos_x < pos_xb + (interval/2):
                is_to_xb = False
                # rospy.loginfo("is_to_xb=False")
                continue
            elif pos_x < pos_xc + (interval/2):
                is_to_xc = False
                # rospy.loginfo("is_to_xc=False")
                continue

        x_line = None   # xa,xb,xcいずれかのリストが入る
        # NOTE 優先的にxcに移動する
        if is_to_xc:
            x_line = waypoints["xc"]
            rospy.loginfo("select next waypoint_xc")
        elif is_to_xb:
            x_line = waypoints["xb"]
            rospy.loginfo("select next waypoint_xb")
        elif is_to_xa:
            x_line = waypoints["xa"]
            rospy.loginfo("select next waypoint_xa")
        else:
            # a,b,cいずれにも移動できない場合
            x_line = waypoints["xb"]
            rospy.loginfo("select default waypoint")

        return x_line[current_stp]

    def execute_task1(self):
        """
        task1を実行する
        """
        rospy.loginfo("#### start Task 1 ####")
        hsr_position = [
            ("near_long_table_l", "look_at_near_floor"),
            ("tall_table", "look_at_tall_table"),
            ("long_table_r", "look_at_tall_table"),
        ]

        # self.pull_out_trofast(0.178, 0.8, 0.75, -90, -100, 0)

        total_cnt = 0
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

                # 検出した全物体を取得
                detected_objs = self.get_latest_detection().bboxes
                
                # 「床にある」と判断した物体候補のリスト
                floor_objects_info = []

                # Y座標（奥行き）のしきい値
                # exec_graspable_method の graspable_y (1.85) と同じ値
                GRASPABLE_Y_THRESHOLD = 1.85 

                for obj in detected_objs:
                    # 物体の3D座標を取得
                    grasp_pos = self.get_grasp_coordinate(obj)
                    if grasp_pos is None:
                        rospy.logwarn("Failed to get coordinate for [%s]", obj.label)
                        break

                    # フィルター：Y座標（奥行き）チェック
                    if grasp_pos.y >= GRASPABLE_Y_THRESHOLD:
                        rospy.loginfo("Ignoring object [%s] (Too far: Y=%.2f)", obj.label, grasp_pos.y)
                        continue # 壁の奥など、奥すぎると判断し、無視

                    # フィルターを通過した物体のみ候補リストに追加
                    score = self.calc_score_bbox(obj)
                    floor_objects_info.append({
                        "bbox": obj, 
                        "score": score, 
                        "label": obj.label, 
                        "pos": grasp_pos  # 3D座標も保存
                    })

                # 候補リストをスコアの高い順にソート
                floor_objects_info.sort(key=lambda x: x["score"], reverse=True)

                # 最終的な把持対象を決定
                if not floor_objects_info:
                    # 候補が一つもなかった
                    rospy.logwarn("Cannot find graspable object on the floor in this view.")
                    continue # 次の検出試行へ

                # 最もスコアの高い物体を選択
                best_obj_info = floor_objects_info[0]
                grasp_pos = best_obj_info["pos"]
                label = best_obj_info["label"]
                # TODO ラベル名を確認するためにコメントアウトを外す
                rospy.loginfo("grasp the " + label)

                # 把持対象がある場合は把持関数実施
                self.change_pose("grasp_on_table")
                
                # 座標チェックは完了しているので、exec_graspable_method を実行
                if not self.exec_graspable_method(grasp_pos, label):
                    rospy.logwarn("exec_graspable_method returned False for [%s]", label)
                    continue
                
                self.change_pose("all_neutral")

                # 1. ラベルからカテゴリと配置場所を取得
                category, place_name = self.get_placement_info(label)
                #place_name = "bin_a"

                # 2. 常に "put_in_bin" の姿勢を使う（テスト用）
                into_pose = "put_in_bin" 

                # 3. 取得した配置場所(place_name)と、固定の姿勢(into_pose)で物体を置く
                self.put_in_place(place_name, into_pose)
                """
                # 1. ラベルからカテゴリと配置場所を取得
                category, place_name = self.get_placement_info(label)

                # 2. カテゴリに応じた投入姿勢を決定
                # (これは仮の実装です。場所ごとに適切な姿勢を指定する必要があります)
                if place_name in ["bin_a_place", "bin_b_place"]:
                    into_pose = "put_in_bin"
                elif place_name in ["tray_a_place", "tray_b_place", "container_a_place", "container_b_place"]:
                    # (例: "put_on_tray"という姿勢をposes.jsonで定義)
                    into_pose = "put_on_tray" 
                elif place_name in ["drawer_left_place", "drawer_top_place", "drawer_bottom_place"]:
                    # (例: "put_in_drawer"という姿勢をposes.jsonで定義)
                    into_pose = "put_in_drawer" 
                else:
                    into_pose = "put_in_bin" # デフォルト

                # 3. 取得した配置場所(place_name)と姿勢(into_pose)を使って物体を置く
                # (注意: 'put_in_bin' 以外は、対応する場所と姿勢を別途定義する必要があります)
                self.put_in_place(place_name, into_pose)
                # binに入れる
                """
                """
                if total_cnt % 2 == 0:  
                    self.put_in_place("bin_a_place", "put_in_bin")
                else:  
                    self.put_in_place("bin_b_place", "put_in_bin")
                total_cnt += 1
                """

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
        #self.execute_task2b()


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

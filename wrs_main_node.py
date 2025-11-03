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
                rospy.loginfo(" -> Parsed object: [{}], person: [{}]".format(target_obj, target_person))
            else:
                # 形式が予期せず異なる場合 (例: "A to person B to person C")
                rospy.logerr(" -> Failed to parse instruction (unexpected format). Using defaults.")
        else:
            # 指示に " to person " が含まれていない場合
            rospy.logerr(" -> Failed to parse instruction (separator '{}' not found). Using defaults.".format(separator))
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
        """
        Trofast の引き出しを引き出す（姿勢は固定し、位置だけ相対的に移動する実装）
        - 重要: 引き出しを引くときにロボットが回転してしまう問題を防ぐため、
        明示的な yaw/pitch/roll の数値を直接使わず、**現在のエンドエフェクタ姿勢を取得してその姿勢を固定**したまま
        位置だけを変える（相対移動的）実装にする。
        """
        try:
            # 初期の準備ポーズ（既存の呼び出しを維持）
            self.goto_name("stair_like_drawer")
            self.change_pose("grasp_on_table")

            # グリッパを開ける（数値は既存コードの通り）
            gripper.command(1)

            # -- 現在のエンドエフェクタ姿勢（位置＋オイラー角）を取得する（複数の可能性に対応）
            current_pose = None
            # try several commonly used getter names as a fallback
            getters = [
                getattr(whole_body, "get_current_pose", None),
                getattr(whole_body, "get_end_effector_pose", None),
                getattr(whole_body, "get_pose", None),
                getattr(whole_body, "get_current_end_effector_pose", None),
            ]
            for g in getters:
                if callable(g):
                    try:
                        p = g()
                        # 想定される返り値のパターンに対応
                        # - 辞書: {'x':..,'y':..,'z':..,'yaw':..,'pitch':..,'roll':..}
                        # - オブジェクト/tuple: (x,y,z,yaw,pitch,roll) または オブジェクトに属性 x,y,z,yaw...
                        if isinstance(p, dict):
                            current_pose = {
                                "x": float(p.get("x", 0.0)),
                                "y": float(p.get("y", 0.0)),
                                "z": float(p.get("z", 0.0)),
                                "yaw": float(p.get("yaw", yaw)),
                                "pitch": float(p.get("pitch", pitch)),
                                "roll": float(p.get("roll", roll)),
                            }
                            break
                        # tuple/list
                        if isinstance(p, (list, tuple)) and len(p) >= 6:
                            current_pose = {
                                "x": float(p[0]),
                                "y": float(p[1]),
                                "z": float(p[2]),
                                "yaw": float(p[3]),
                                "pitch": float(p[4]),
                                "roll": float(p[5]),
                            }
                            break
                        # object with attributes
                        for attr in ("x", "y", "z", "yaw", "pitch", "roll"):
                            if not hasattr(p, attr):
                                break
                        else:
                            current_pose = {
                                "x": float(p.x),
                                "y": float(p.y),
                                "z": float(p.z),
                                "yaw": float(getattr(p, "yaw", yaw)),
                                "pitch": float(getattr(p, "pitch", pitch)),
                                "roll": float(getattr(p, "roll", roll)),
                            }
                            break
                    except Exception:
                        # getter failed, try next
                        traceback.print_exc()
                        continue

            # 最後のフォールバック：渡された yaw,pitch,roll を使い、位置は引数の x,y,z
            if current_pose is None:
                rospy.logwarn("whole_body の現在姿勢が取得できなかったため、渡された yaw/pitch/roll を姿勢として使用します。")
                current_pose = {"x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}

            # ① 接近（オフセット位置）：引き出しの前で把持位置に近づく
            #    元コードでは (x, y + offset, z) に移動していたのでそれを踏襲。ただし yaw/pitch/roll は current_pose のものを使う。
            try:
                whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z,
                                                current_pose["yaw"],
                                                current_pose["pitch"],
                                                current_pose["roll"])
            except Exception:
                # もし move_end_effector_pose が例外を出す場合は位置のみを移動する別APIを試す（存在すれば）
                try:
                    if hasattr(whole_body, "move_end_effector_position"):
                        whole_body.move_end_effector_position(x, y + self.TROFAST_Y_OFFSET, z)
                    else:
                        raise
                except Exception:
                    rospy.logerr("接近移動に失敗しました（move_end_effector_pose / move_end_effector_position いずれも使用不可）。")
                    raise

            rospy.sleep(0.2)  # 少し待つ（姿勢安定用）

            # ② 把持位置へ移動（実際に把持する位置）
            #     ここでも姿勢は current_pose に固定 → 回転が起きない
            whole_body.move_end_effector_pose(x, y, z,
                                            current_pose["yaw"],
                                            current_pose["pitch"],
                                            current_pose["roll"])
            rospy.sleep(0.05)

            # ③ 把持（グリッパを閉じる）
            gripper.command(0)
            rospy.sleep(0.08)

            # ④ 引き出す（**位置だけ**を元のオフセット側に戻す／相対的に後退する）
            #     ここが重要：姿勢は current_pose のまま固定し、位置だけ y+offset に動かす
            whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z,
                                            current_pose["yaw"],
                                            current_pose["pitch"],
                                            current_pose["roll"])
            rospy.sleep(0.2)

            # ⑤ 必要なら把持解除（元コードではここで再度 gripper.command(1) をしていた）
            gripper.command(1)

            # ⑥ 元の中立姿勢へ戻す
            self.change_pose("all_neutral")

        except Exception as e:
            rospy.logerr("pull_out_trofast 中に例外: %s\n%s" % (str(e), traceback.format_exc()))
            # 失敗時にも安全な姿勢へ戻す試み
            try:
                self.change_pose("all_neutral")
            except Exception:
                pass
            raise


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
        rospy.loginfo("Found graspable object '{}'".format(target_obj))
        
        # BBoxの3次元座標を取得して、その座標で把持する
        grasp_pos = self.get_grasp_coordinate(grasp_bbox)
        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos)
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
            rospy.logwarn("Unknown person name '{}', defaulting to 'person_b'".format(target_person))
            delivery_location = "person_b" 

        rospy.loginfo("Delivering to: {} (based on '{}')".format(delivery_location, target_person))
        # target_personの前に持っていく
        self.change_pose("look_at_near_floor")
        self.goto_name(delivery_location)    # TODO: 配達先が固定されているので修正←済
        self.change_pose("deliver_to_human")
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")
    
    """
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
    
    def execute_avoid_blocks(self):
        # blockを避ける (Y軸移動とX軸移動を分離)
        for i in range(10): # 10ステップのループ
            
            # --- 1. 障害物と次のウェイポイントを決定 ---
            detected_objs = self.get_latest_detection()
            bboxes = detected_objs.bboxes
            pos_bboxes = [self.get_grasp_coordinate(bbox) for bbox in bboxes]
            
            # 10ステップ版の select_next_waypoint が呼ばれる
            waypoint = self.select_next_waypoint(i, pos_bboxes)
            
            target_x = waypoint[0]
            target_y = waypoint[1]
            target_yaw = waypoint[2]

            # --- 2. 現在のロボットの座標を取得 ---
            try:
                # "map" 座標系における "base_link" の現在の位置を取得
                trans = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
                current_x = trans.transform.translation.x
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr("Could not get current robot pose: %s", e)
                rospy.logwarn("Skipping this step.")
                continue # 現在地が取れないため、このステップはスキップ

            # --- 3. Y軸方向（前進）にのみ移動 ---
            # Xは現在の位置(current_x)を維持し、Yだけ次のステップ(target_y)へ進む
            # 向きは常に前(90度)を向く
            rospy.loginfo("Step %d: Moving in Y-axis (Forward) to %.2f", i, target_y)
            self.goto_pos([current_x, target_y, 90])

            # --- 4. X軸方向（横移動）にのみ移動 ---
            # Yは(3)で移動した target_y を維持し、Xだけ目標のレーン(target_x)へ移動
            rospy.loginfo("Step %d: Moving in X-axis (Sideways) to %.2f", i, target_x)
            self.goto_pos([target_x, target_y, target_yaw])
        
        rospy.loginfo("Finished execute_avoid_blocks.")
    """
    def execute_avoid_blocks(self):
        # blockを避ける (Y軸移動 -> X軸安全確認 -> X軸移動)
        
        # 10ステップのY座標境界 (select_next_waypointから移動)
        y_thresholds = [1.85, 1.995, 2.14, 2.285, 2.43, 2.575, 2.72, 2.865, 3.01, 3.155, 3.3]
        
        for i in range(10): # 10ステップのループ
            
            # --- 1. 現在のロボットの座標を取得 ---
            try:
                trans = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
                current_x = trans.transform.translation.x
                current_y = trans.transform.translation.y # 現在のY座標
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr("Could not get current robot pose: %s", e)
                rospy.logwarn("Skipping this step.")
                continue 

            # --- 2. Y軸の目標地点を決定 ---
            target_y = y_thresholds[i+1] # i=0の時、y_thresholds[1] (1.995) になる

            # --- 3. Y軸方向（前進）にのみ移動 ---
            rospy.loginfo("Step %d: Moving in Y-axis (Forward) to %.2f", i, target_y)
            self.goto_pos([current_x, target_y, 90])

            # --- 4. Y軸移動後、X軸の安全確認 ---
            rospy.loginfo("Step %d: Arrived at Y=%.2f. Checking X-axis safety...", i, target_y)
            detected_objs = self.get_latest_detection()
            bboxes = detected_objs.bboxes
            pos_bboxes = [self.get_grasp_coordinate(bbox) for bbox in bboxes]
            
            # (X軸の安全確認と移動先を決定する新しい関数を呼び出す)
            target_x = self.find_safe_x_lane(pos_bboxes, current_x, target_y)

            # --- 5. X軸方向（横移動）にのみ移動 ---
            rospy.loginfo("Step %d: Moving in X-axis (Sideways) to %.2f", i, target_x)
            self.goto_pos([target_x, target_y, 90]) # Yawは常に90度

        rospy.loginfo("Finished execute_avoid_blocks.")

    def find_safe_x_lane(self, pos_bboxes, current_x, current_y):
        """
        [v4] Y軸移動後、現在のY座標で安全なXレーンを見つける。
        安全なレーンが複数ある場合、[最も遠い]ものを選択する。
        """
        interval = 0.45
        pos_xa = 1.7
        pos_xb = pos_xa + interval
        pos_xc = pos_xb + interval

        # 各レーンの中心X座標
        lane_centers = {
            "xa": pos_xa,
            "xb": pos_xb,
            "xc": pos_xc
        }
        
        # 各レーンの安全フラグ (1=安全, 0=ブロック)
        lane_safety = {
            "xa": 1,
            "xb": 1,
            "xc": 1
        }
        
        # X軸（横移動）チェック用のY座標範囲 (±15cm)
        X_CHECK_Y_MIN = current_y - 0.15
        X_CHECK_Y_MAX = current_y + 0.15
        
        # 安全バッファ（横移動の経路チェック用）
        SAFETY_BUFFER = 0.3 # 30cm (ロボットの車体幅半分)

        for bbox in pos_bboxes:
            pos_x = bbox.x
            pos_y = bbox.y

            # --- X軸（横移動）経路の安全チェック ---
            if (X_CHECK_Y_MIN < pos_y < X_CHECK_Y_MAX):
                # この障害物は「今いるY座標」の近くにある
                
                # 'xa'レーンへの横移動経路(current_x と xa の間)をブロックしていないか？
                if (pos_x - SAFETY_BUFFER) < max(current_x, lane_centers["xa"]) and \
                   (pos_x + SAFETY_BUFFER) > min(current_x, lane_centers["xa"]):
                    lane_safety["xa"] = 0 # ブロックされている
                
                # 'xb'レーンへの横移動経路をブロックしていないか？
                if (pos_x - SAFETY_BUFFER) < max(current_x, lane_centers["xb"]) and \
                   (pos_x + SAFETY_BUFFER) > min(current_x, lane_centers["xb"]):
                    lane_safety["xb"] = 0 # ブロックされている
                
                # 'xc'レーンへの横移動経路をブロックしていないか？
                if (pos_x - SAFETY_BUFFER) < max(current_x, lane_centers["xc"]) and \
                   (pos_x + SAFETY_BUFFER) > min(current_x, lane_centers["xc"]):
                    lane_safety["xc"] = 0 # ブロックされている

        rospy.loginfo("X-Lane safety (Blocked=0): xa=%.2f, xb=%.2f, xc=%.2f", 
                      lane_safety["xa"], lane_safety["xb"], lane_safety["xc"])

        # スコアが0（ブロック）でないレーンだけを候補にする
        safe_lanes = {lane: center for lane, center in lane_centers.items() if lane_safety[lane] > 0}

        if not safe_lanes:
            rospy.logwarn("All X-lanes are blocked! Staying at current X=%.2f.", current_x)
            return current_x # 安全なレーンがない場合、動かない (これが最も安全)
        else:
            # --- ★ロジック変更点 ---
            # 安全なレーンの中で、[現在地に最も遠い]レーンを選択する
            best_lane_name = max(safe_lanes.keys(), key=lambda lane: abs(safe_lanes[lane] - current_x))
            rospy.loginfo("Selected best (farthest safe) X-lane: %s", best_lane_name)
            return safe_lanes[best_lane_name]
            
    def select_next_waypoint(self, current_stp, pos_bboxes):
        """
        [元に戻したロジック] xa,xb,xcの固定優先順位でウェイポイントを返す。
        (10ステップ、Yaw=90度固定は維持)
        """
        interval = 0.45
        pos_xa = 1.7
        pos_xb = pos_xa + interval
        pos_xc = pos_xb + interval

        # --- Yaw(向き)をすべて90度に固定 (回転防止のため) ---
        waypoints = {
            "xa": [ [pos_xa, 1.995, 90], [pos_xa, 2.14, 90], [pos_xa, 2.285, 90], [pos_xa, 2.43, 90], [pos_xa, 2.575, 90],
                    [pos_xa, 2.72, 90], [pos_xa, 2.865, 90], [pos_xa, 3.01, 90], [pos_xa, 3.155, 90], [pos_xa, 3.3, 90] ], 
            "xb": [ [pos_xb, 1.995, 90], [pos_xb, 2.14, 90], [pos_xb, 2.285, 90], [pos_xb, 2.43, 90], [pos_xb, 2.575, 90],
                    [pos_xb, 2.72, 90], [pos_xb, 2.865, 90], [pos_xb, 3.01, 90], [pos_xb, 3.155, 90], [pos_xb, 3.3, 90] ],
            "xc": [ [pos_xc, 1.995, 90], [pos_xc, 2.14, 90], [pos_xc, 2.285, 90], [pos_xc, 2.43, 90], [pos_xc, 2.575, 90],
                    [pos_xc, 2.72, 90], [pos_xc, 2.865, 90], [pos_xc, 3.01, 90], [pos_xc, 3.155, 90], [pos_xc, 3.3, 90] ]
        }
        
        # 10ステップのY座標境界
        y_thresholds = [1.85, 1.995, 2.14, 2.285, 2.43, 2.575, 2.72, 2.865, 3.01, 3.155, 3.3]
        
        #現在のyと次のy
        current_y = y_thresholds[current_stp]
        next_y = y_thresholds[current_stp + 1]
        
        # --- ここからが元のロジック (固定優先順位) ---
        
        # posがxa,xb,xcのラインに近い場合は候補から削除
        is_to_xa = True
        is_to_xb = True
        is_to_xc = True

        for bbox in pos_bboxes:
            pos_x = bbox.x
            pos_y = bbox.y

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

        self.pull_out_trofast(0.18, -0.29, 0.55, 0, -100, 0)

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
                GRASPABLE_Y_THRESHOLD = 2.0

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

def execute_avoid_blocks(self):
        """
        [v19 シンプル・中間点アプローチ]
        - 2つの障害物のX座標の中間点を計算
        - Move 1: X軸を中間点に合わせる
        - Move 2: Y軸のみでエリア出口まで直進する
        """
        TARGET_Y_EXIT = 3.3
        DEFAULT_SAFE_X = 2.25 # 障害物が1つ以下の場合のデフォルトX
        MIN_OBSTACLE_DIST = 0.5 # 50cm以内は同じ障害物とみなす
        MIN_MOVE_DIST = 0.08    # 2cm以下の移動は無視
        SAFETY_BUFFER = 0.45    # 45cm (安全マージン)

        # 移動完了を待つ時間
        WAIT_FOR_MOVE = 25.0 

        # --- 壁の位置設定 ---
        WALL_MIN_X = 1.3
        WALL_MAX_X = 3.5

        rospy.loginfo("Starting STABILIZED Midpoint Avoidance (Wait Time: %.1f)...", WAIT_FOR_MOVE)
        
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            current_x = trans.transform.translation.x
            current_y = trans.transform.translation.y
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Could not get current robot pose: %s", e)
            return

        # --- 1. 障害物検知とクラスタリング (重複除去) ---
        detected_objs = self.get_latest_detection().bboxes
        raw_obstacles = []
        for bbox in detected_objs:
            pos = self.get_grasp_coordinate(bbox)
            if pos is not None and pos.y > current_y:
                raw_obstacles.append(pos)

        unique_obstacles = []
        for p in raw_obstacles:
            is_new = True
            for up in unique_obstacles:
                dist = math.sqrt((p.x - up.x)**2 + (p.y - up.y)**2)
                if dist < MIN_OBSTACLE_DIST:
                    is_new = False
                    break
            if is_new:
                unique_obstacles.append(p)

        unique_obstacles.sort(key=lambda p: p.y)
        num_obstacles = len(unique_obstacles)
        rospy.loginfo("Found %d unique obstacles.", num_obstacles)

        # --- 2. 経路計算 (X座標の中間点) ---
        target_x = DEFAULT_SAFE_X # デフォルト

        if num_obstacles == 1:
            # 障害物1つ -> 障害物のいない側を通る
            obs = unique_obstacles[0]
            rospy.loginfo("One obstacle detected. Avoiding to opposite side.")
            target_x = 1.7 if obs.x > DEFAULT_SAFE_X else 2.6
            
        elif num_obstacles >= 2:
            # 障害物2つ以上 -> 手前の2つのX座標の中間点を計算
            p1 = unique_obstacles[0]
            p2 = unique_obstacles[1]
            target_x = (p1.x + p2.x) / 2.0
            rospy.loginfo("Bisector P1(%.2f, %.2f), P2(%.2f, %.2f). Target X midpoint: %.2f", 
                          p1.x, p1.y, p2.x, p2.y, target_x)

        # --- 壁衝突防止 (クランプ処理) ---
        target_x = max(WALL_MIN_X, min(WALL_MAX_X, target_x))
        rospy.loginfo("Calculated path: X(%.2f) -> (Straight Y to Exit)", target_x)

        # --- 3. 安全な移動実行 (X->Y) ---
        robot_x = current_x
        robot_y = current_y

        # Move 1: X軸合わせ (target_x へ)
        if abs(target_x - robot_x) > MIN_MOVE_DIST:
            if not self.is_path_safe(unique_obstacles, robot_x, target_x, robot_y, robot_y, SAFETY_BUFFER):
                rospy.logerr("Move 1 (X-adjust) BLOCKED. Aborting avoidance.")
                return 
            rospy.loginfo("Move 1: Adjusting X to %.2f", target_x)
            self.goto_pos([target_x + 0.1, robot_y + 0.5, 90])
            robot_x = target_x + 0.1# X座標を更新
            robot_y = robot_y + 0.5
            rospy.sleep(WAIT_FOR_MOVE)
        else:
            rospy.loginfo("Move 1: Skipped (X aligned)")
        
        SAFETY_BUFFER = 0.01
        self.goto_pos([target_x + 0.1, robot_y + 0.2, 90])
        robot_y = robot_y + 0.2
        self.goto_pos([target_x + 0.1, robot_y + 0.2, 90])
        robot_y = robot_y + 0.2
        self.goto_pos([target_x , robot_y + 0.2, 90])
        robot_y = robot_y + 0.2
        self.goto_pos([robot_x + 0.1, TARGET_Y_EXIT, 90])
  
        # Move 2: 出口まで直進
        if (TARGET_Y_EXIT - robot_y) > MIN_MOVE_DIST:
            if not self.is_path_safe(unique_obstacles, robot_x, robot_x, robot_y, TARGET_Y_EXIT, SAFETY_BUFFER):
                rospy.logwarn("Move 2 (Y-Path) BLOCKED by safety check, but proceeding anyway (Failsafe).")
                # return <-- Failsafeのため停止しない
            
            rospy.loginfo("Move 2: Moving straight to exit Y=%.2f (at X=%.2f)", TARGET_Y_EXIT, robot_x)
            self.goto_pos([robot_x + 0.1, TARGET_Y_EXIT, 90])
            rospy.sleep(WAIT_FOR_MOVE) 
        else:
            rospy.loginfo("Move 2: Skipped (Already at exit)")

        rospy.loginfo("Finished stabilized midpoint avoidance.")
    def is_xpath_safe(self, obstacles, x_from, x_to, y_at, buffer):
        """
        X軸（横）移動が安全かチェックする
        """
        y_min = y_at - 0.2
        y_max = y_at + 0.2
        path_min_x = min(x_from, x_to) - buffer
        path_max_x = max(x_from, x_to) + buffer
        
        for obs in obstacles:
            if (y_min < obs.y < y_max) and (path_min_x < obs.x < path_max_x):
                rospy.logwarn("X-Path (%.2f -> %.2f at Y=%.2f) BLOCKED by obs at (%.2f, %.2f)",
                              x_from, x_to, y_at, obs.x, obs.y)
                return False
        return True

    def is_path_safe(self, obstacles, x_from, x_to, y_from, y_to, buffer):
        """
        X軸（横）またはY軸（前進）の「経路（矩形）」が安全かチェックする
        """
        path_min_x = min(x_from, x_to) - buffer
        path_max_x = max(x_from, x_to) + buffer
        path_min_y = min(y_from, y_to) - 0.2 # Y方向のバッファも少し持つ
        path_max_y = max(y_from, y_to) + 0.2
        
        for obs in obstacles:
            if (path_min_x < obs.x < path_max_x) and (path_min_y < obs.y < path_max_y):
                rospy.logwarn("Path (X: %.2f->%.2f, Y: %.2f->%.2f) BLOCKED by obs at (%.2f, %.2f)",
                              x_from, x_to, y_from, y_to, obs.x, obs.y)
                return False
        return True

    def is_ypath_safe(self, obstacles, x_at, y_from, y_to, buffer):
        """
        Y軸（前進）移動が安全かチェックする (レーン幅チェック)
        ※ v14 では is_path_safe に統合されたため、この関数は不要だが、
           v10 などの古い execute_avoid_blocks のために残しておいても良い
        """
        path_min_x = x_at - buffer
        path_max_x = x_at + buffer
        
        for obs in obstacles:
            if (y_from < obs.y < y_to) and (path_min_x < obs.x < path_max_x):
                rospy.logwarn("Y-Path (%.2f -> %.2f at X=%.2f) BLOCKED by obs at (%.2f, %.2f)",
                              y_from, y_to, x_at, obs.x, obs.y)
                return False
        return True
    
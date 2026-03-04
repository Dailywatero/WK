#!/usr/bin/env python3
import rospy
import actionlib
import threading
import time
import math
import tf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
import sys
import os

# 添加 g1_action.py 所在目录到 Python 路径
sys.path.append("/home/unitree/WK/unitree_sdk2_python/example/g1/high_level")

# 导入 G1ActionController 类
from g1_action import G1ActionController


current_global_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
pose_lock = threading.Lock()

class RobotController:
    def __init__(self, network_interface):
        rospy.loginfo("正在初始化语音和动作系统...")
        ChannelFactoryInitialize(0, network_interface)
        self.audio_client = AudioClient()
        self.audio_client.Init()
        self.audio_client.SetTimeout(10.0)
        self.arm_client = G1ArmActionClient()
        self.arm_client.Init()
        self._wakeup_audio()
        self.global_plan_length = 0
        self.loco_client = LocoClient()
        self.loco_client.SetTimeout(10.0)
        self.loco_client.Init()
        rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self._path_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.loginfo("✅ 语音和动作系统初始化完成")

    def _path_callback(self, msg: Path):
        self.global_plan_length = len(msg.poses)

    def _wakeup_audio(self):
        rospy.loginfo("🔊 正在唤醒音频硬件...")
        for i in range(10):
            try:
                self.audio_client.GetVolume()
                rospy.loginfo("✅ 音频服务已连接")
                break
            except:
                rospy.logwarn(f"⏳ 等待音频服务... ({i+1}/10)")
                time.sleep(1)
        self.audio_client.SetVolume(100)
        time.sleep(0.5)
        rospy.loginfo("✅ 音频硬件唤醒完成")

    def speak(self, text):
        try:
            rospy.loginfo(f"🔊 说: {text}")
            self.audio_client.TtsMaker(text, 0)
        except Exception as e:
            rospy.logerr(f"语音播放失败: {e}")

    def perform_interaction(self, text, action_id):
        rospy.loginfo(f"🤖 执行动作 ID: {action_id}")
        try:
            self.arm_client.ExecuteAction(action_id)
            time.sleep(2.0)
        except Exception as e:
            rospy.logerr(f"动作执行失败: {e}")

        rospy.loginfo(f"🎤 播放: {text}")
        self.speak(text)
        
        estimated_speech_time = len(text) * 0.195
        rospy.loginfo(f"⏳ 预计讲解时长: {estimated_speech_time:.1f} 秒，正在等待讲解结束...")
        time.sleep(estimated_speech_time)

        rospy.loginfo("🔄 讲解结束，正在复位手臂...")
        try:
            self.arm_client.ExecuteAction(99)
            time.sleep(3.0)
        except Exception as e:
            rospy.logerr(f"复位失败: {e}")

    # 【新增】手动原地旋转修正航向函数
    def rotate_to_yaw(self, target_yaw, listener):
        rospy.loginfo(f"🔄 开始原地旋转修正航向至: {math.degrees(target_yaw):.1f}°")
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            try:
                # 获取当前姿态
                (trans, rot) = listener.lookupTransform("/map", "/base_link", rospy.Time(0))
                current_yaw = euler_from_quaternion(rot)[2]
                
                # 计算角度差
                yaw_diff = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
                
                # 如果角度误差小于 0.1 弧度 (约 6度)，认为对齐成功
                if abs(yaw_diff) < 0.1:
                    rospy.loginfo("✅ 航向对齐完成")
                    break
                
                # 简单 P 控制器计算角速度
                # 最大旋转速度限制在 0.6 rad/s，保证平稳
                cmd_wz = max(-0.6, min(0.6, yaw_diff * 1.5))
                
                # 发布旋转指令 (vx=0, vy=0, wz=cmd_wz)
                twist_msg = Twist()
                twist_msg.angular.z = cmd_wz
                self.cmd_vel_pub.publish(twist_msg)
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
            
            rate.sleep()
        
        # 停止旋转
        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.1)


def set_fast_params():
    rospy.set_param('/move_base/TebLocalPlannerROS/max_vel_x', 0.8)
    rospy.set_param('/move_base/TebLocalPlannerROS/max_vel_theta', 0.8)
    rospy.set_param('/move_base/TebLocalPlannerROS/acc_lim_x', 0.8)
    rospy.set_param('/move_base/TebLocalPlannerROS/acc_lim_theta', 0.7)
    rospy.set_param('/move_base/TebLocalPlannerROS/path_distance_bias', 60.0)
    rospy.set_param('/move_base/TebLocalPlannerROS/goal_distance_bias', 20.0)
    rospy.set_param('/move_base/TebLocalPlannerROS/xy_goal_tolerance', 0.3)
    rospy.set_param('/move_base/TebLocalPlannerROS/yaw_goal_tolerance', 0.3)
    rospy.set_param('/move_base/TebLocalPlannerROS/min_turning_radius', 0.0)
    rospy.set_param('/move_base/TebLocalPlannerROS/weight_optimaltime', 1.0)
    rospy.loginfo("🚀 切换到快速巡航模式")

def set_slow_params():
    rospy.set_param('/move_base/TebLocalPlannerROS/max_vel_x', 0.6)
    rospy.set_param('/move_base/TebLocalPlannerROS/max_vel_theta', 1.0)
    rospy.set_param('/move_base/TebLocalPlannerROS/acc_lim_x', 0.5)
    rospy.set_param('/move_base/TebLocalPlannerROS/acc_lim_theta', 0.8)
    rospy.set_param('/move_base/TebLocalPlannerROS/min_turning_radius', 0.0)
    rospy.set_param('/move_base/TebLocalPlannerROS/max_vel_x_backwards', 0.0)
    rospy.set_param('/move_base/TebLocalPlannerROS/xy_goal_tolerance', 0.2)
    rospy.set_param('/move_base/TebLocalPlannerROS/yaw_goal_tolerance', 0.2)
    rospy.loginfo("🐌 切换到精准调整模式")


def force_robot_stop(robot_controller_instance):
    rospy.logwarn("🛑 强制接管控制...")
    try:
        robot_controller_instance.loco_client.StopMove()
        stop_msg = Twist()
        for _ in range(10):
            robot_controller_instance.cmd_vel_pub.publish(stop_msg)
            time.sleep(0.02)
        rospy.loginfo("🛑 SDK StopMove 指令已发送")
    except Exception as e:
        rospy.logerr(f"SDK 停止指令发送失败: {e}")
    time.sleep(0.1)

def navigate_to_waypoints(waypoints, robot_controller):
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("等待move_base服务器启动...")
    client.wait_for_server()

    clear_costmaps_service = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
    
    # 公共参数初始化
    rospy.set_param('/move_base/DWAPlannerROS/yaw_goal_tolerance', 0.4)
    rospy.set_param('/move_base/DWAPlannerROS/max_rot_vel', 0.6)
    rospy.set_param('/move_base/DWAPlannerROS/min_rot_vel', 0.2)
    rospy.set_param('/move_base/DWAPlannerROS/acc_lim_theta', 0.2)
    rospy.set_param('/move_base/DWAPlannerROS/path_distance_bias', 40.0)
    rospy.set_param('/move_base/DWAPlannerROS/goal_distance_bias', 15.0)
    rospy.set_param('/move_base/planner_patience', 18.0)
    rospy.set_param('/move_base/planner_frequency', 1.0)

    listener = tf.TransformListener()
    base_frame = "/base_link" 
    map_frame = "/map"
    
    # 【修改1】放宽停止角度容差，主要依靠距离判定
    STOP_YAW_TOLERANCE = 3.14  # 几乎任何角度都允许停止

 # =========== 【新增】在循环外创建一次 ===========
    rospy.loginfo("初始化动作控制器...")
    action_controller = G1ActionController(network_interface, skip_channel_init=True)

    for idx, waypoint in enumerate(waypoints):
        rospy.loginfo("⏳ 获取机器人在地图中的真实初始位置...")
        try:
            listener.waitForTransform(map_frame, base_frame, rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = listener.lookupTransform(map_frame, base_frame, rospy.Time(0))
            with pose_lock:
                current_global_pose["x"] = trans[0]
                current_global_pose["y"] = trans[1]
            rospy.loginfo(f"📍 初始全局位置: ({current_global_pose['x']:.2f}, {current_global_pose['y']:.2f})")
        except Exception as e:
            rospy.logerr(f"❌ 无法获取 TF 变换: {e}")
            continue

        start_dist = math.sqrt(
            (waypoint["x"] - current_global_pose["x"]) ** 2 + 
            (waypoint["y"] - current_global_pose["y"]) ** 2
        )
        
        # 【修改2】停止距离设置
        if start_dist < 1.0:
            STOP_DISTANCE = 0.4
        else:
            STOP_DISTANCE = 0.7
            
        # 【修改3】提前切换到慢速的距离 (从 1.0 改为 1.5)
        SLOWDOWN_DISTANCE = 1.5

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = waypoint["x"]
        goal.target_pose.pose.position.y = waypoint["y"]
        goal.target_pose.pose.position.z = 0.0

        q = quaternion_from_euler(0, 0, waypoint["yaw"])
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        rospy.loginfo(f"发送第{idx+1}个目标: ({waypoint['x']}, {waypoint['y']}, yaw: {waypoint['yaw']:.2f})")
        
        set_fast_params()
        client.send_goal(goal)

        start_time = time.time()
        last_spoke_time = 0.0
        speak_interval = 10.0
        obstacle_start_time = 0.0 
        map_cleared = False
        is_fast_mode = True
        slowdown_switched = False
        
        while not rospy.is_shutdown():
            state = client.get_state()
            elapsed = time.time() - start_time
            now = time.time()

            # 【修改4】到达判定逻辑优化
            def handle_arrival(reason):
                rospy.loginfo(f"✅ {reason}：触发第{idx+1}个点停止流程")
                client.cancel_goal()
                force_robot_stop(robot_controller)
                
                # 到达后，检查是否需要修正航向
                try:
                    (trans, rot) = listener.lookupTransform(map_frame, base_frame, rospy.Time(0))
                    current_yaw = euler_from_quaternion(rot)[2]
                    yaw_diff = math.atan2(math.sin(waypoint["yaw"] - current_yaw), math.cos(waypoint["yaw"] - current_yaw))
                    
                    # 如果航向偏差大，进行原地旋转修正
                    if abs(yaw_diff) > 0.15: # 约 8 度
                        robot_controller.rotate_to_yaw(waypoint["yaw"], listener)
                except:
                    pass
                
                say_text = waypoint.get("say_text", "你好")
                action_id = waypoint.get("action_id", 25)
                robot_controller.perform_interaction(say_text, action_id)
                
                # 【新增】交互循环：按 Enter 继续，按 A 执行自定义动作
                while True:
                    print(f"\n{'='*50}")
                    print(f"📍 已到达第 {idx+1}/{len(waypoints)} 个导航点")
                    print(f"{'='*50}")
                    print("按键选项：")
                    print("  [Enter] - 前往下一个导航点")
                    print("  [A]     - 执行自定义动作 (比耶)")
                    print(f"{'='*50}")
                    
                    user_input = input("请选择: ").strip().upper()
                    
                    if user_input == 'A':
                        print("🤖 正在执行自定义动作...")
                        try:
                            # 创建动作控制器并执行
                            action_controller.run_action_sequence()
                            print("✅ 自定义动作执行完毕")
                        except Exception as e:
                            rospy.logerr(f"执行自定义动作失败: {e}")
                        # 执行完后继续循环，等待下一个指令
                    elif user_input == '':
                        print("🚀 准备前往下一站...")
                        break  # 跳出循环，继续导航
                
                rospy.loginfo("🎯 准备前往下一站")
                return True


            if state == actionlib.GoalStatus.ACTIVE:
                try:
                    (trans, rot) = listener.lookupTransform(map_frame, base_frame, rospy.Time(0))
                    cur_x = trans[0]
                    cur_y = trans[1]
                    current_yaw = euler_from_quaternion(rot)[2]
                    
                    with pose_lock:
                        current_global_pose["x"] = cur_x
                        current_global_pose["y"] = cur_y
                        current_global_pose["yaw"] = current_yaw

                    distance_to_goal = math.sqrt(
                        (waypoint["x"] - cur_x) ** 2 + 
                        (waypoint["y"] - cur_y) ** 2
                    )
                    
                    yaw_diff = math.atan2(
                        math.sin(waypoint["yaw"] - current_yaw), 
                        math.cos(waypoint["yaw"] - current_yaw)
                    )

                    # 【修改3生效处】更早切换慢速
                    if is_fast_mode and distance_to_goal < SLOWDOWN_DISTANCE and not slowdown_switched:
                        rospy.loginfo(f"📏 距离目标 {distance_to_goal:.2f}m，切换到慢速模式")
                        set_slow_params()
                        is_fast_mode = False
                        slowdown_switched = True
                    
                    # 【修改1生效处】只要距离够近就停止，不再强求角度
                    if elapsed > 0.8 and distance_to_goal < STOP_DISTANCE:
                        if handle_arrival(f"物理距离达标 (Dist: {distance_to_goal:.2f}m)"):
                            break

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    pass

                if elapsed > 0.8: 
                    if robot_controller.global_plan_length == 0:
                        if obstacle_start_time == 0.0:
                            obstacle_start_time = now
                        
                        if now - obstacle_start_time > 1.5: 
                            if now - last_spoke_time > speak_interval:
                                robot_controller.speak("请让一让我")
                                last_spoke_time = now
                            
                            if (not map_cleared):
                                rospy.loginfo("🧹 尝试清除代价地图...")
                                try:
                                    clear_costmaps_service()
                                    map_cleared = True
                                except Exception as e:
                                    rospy.logerr(f"清除代价地图失败: {e}")
                    else:
                        if obstacle_start_time != 0.0:
                            rospy.loginfo("✅ 路径已恢复")
                            obstacle_start_time = 0.0
                            map_cleared = False

            if state == actionlib.GoalStatus.SUCCEEDED:
                if handle_arrival("Move_base 判定到达"):
                    break

            if state == actionlib.GoalStatus.ABORTED:
                rospy.logwarn(f"⚠️ 第{idx+1}个目标导航失败，跳过")
                robot_controller.speak("无法到达该目标位置，即将前往下一位置")
                time.sleep(3.0)
                break

            rospy.sleep(0.01) 

if __name__ == '__main__':
    try:
        rospy.init_node('multi_waypoint_nav')

        import sys
        if len(sys.argv) < 2:
            rospy.logerr("使用方法: rosrun your_package multi_nav_action_audio.py network_interface")
            sys.exit(1)

        network_interface = sys.argv[1]
        robot_controller = RobotController(network_interface)

        rospy.sleep(1.0)
        robot_controller.speak("启动成功")

        waypoints = [
            {"x": 5.33, "y": -1.30, "yaw": 1.09 ,"action_id": 31,"say_text": "3333"},
            {"x": -0.15, "y":-1.62, "yaw": -0.83, "action_id":31,"say_text": "7777"},
            {"x": 5.13, "y": 7.04, "yaw": 2.77, "action_id": 31, "say_text": "8888"},
            {"x": -1.78, "y": 6.37, "yaw": -2.98, "action_id": 31, "say_text": "2222"},
        ]

        navigate_to_waypoints(waypoints, robot_controller)

    except rospy.ROSInterruptException:
        rospy.loginfo("导航脚本被中断")

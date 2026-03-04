#!/usr/bin/env python3
"""
G1 动作控制器 (Python版)
- 功能：在机器人站立时控制上半身做动作（如比耶、抬手）
- 安全：自动锁定腿部关节，保持站立姿态
- 用法：python3 g1_action.py eth0
"""

import sys
import time
import math
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # 默认初始化函数


# ==================== 常量定义 ====================
TOPIC_ARM_SDK = "rt/arm_sdk"
TOPIC_LOWSTATE = "rt/lowstate"
CTRL_DT = 0.02  # 20ms 控制周期

# 关节索引 (与 C++ 版本一致)
class JointIndex:
    # 左腿
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnkle = 4
    LeftAnkleRoll = 5
    # 右腿
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnkle = 10
    RightAnkleRoll = 11
    # 腰部
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    # 左臂
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbowPitch = 18
    LeftElbowRoll = 19
    # 右臂
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbowPitch = 25
    RightElbowRoll = 26
    # 权重控制
    NotUsedJoint = 29


# ==================== 动作控制器类 ====================
class G1ActionController:
    def __init__(self, network_interface, skip_channel_init=False):
        print("=" * 50)
        print("Initializing G1 Action Controller...")
        print("=" * 50)
        
        # 【修改】支持跳过 ChannelFactory 初始化（避免重复初始化冲突）
        if not skip_channel_init:
            ChannelFactoryInitialize(0, network_interface)
        
        # 创建发布者
        self.publisher = ChannelPublisher(TOPIC_ARM_SDK, LowCmd_)
        self.publisher.Init()
        
        # 创建订阅者
        self.subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.subscriber.Init(self._state_callback)
        
        # 【关键】状态变量初始化
        self.current_state = None
        self.weight = 0.0
        self.crc = CRC()
        
        # 【关键】可调参数
        self.kp_upper = 60.0    # 上半身刚度
        self.kd_upper = 1.5     # 上半身阻尼
        self.kp_lower = 80.0    # 下半身刚度 (站立保持)
        self.kd_lower = 2.0     # 下半身阻尼
        self.max_joint_velocity = 0.5  # 最大关节速度
        
        # 【关键】关节分组
        self.arm_joints = [
            JointIndex.LeftShoulderPitch, JointIndex.LeftShoulderRoll,
            JointIndex.LeftShoulderYaw, JointIndex.LeftElbowPitch,
            JointIndex.LeftElbowRoll,
            JointIndex.RightShoulderPitch, JointIndex.RightShoulderRoll,
            JointIndex.RightShoulderYaw, JointIndex.RightElbowPitch,
            JointIndex.RightElbowRoll,
            JointIndex.WaistYaw, JointIndex.WaistRoll, JointIndex.WaistPitch
        ]
        
        self.leg_joints = [
            JointIndex.LeftHipPitch, JointIndex.LeftHipRoll,
            JointIndex.LeftHipYaw, JointIndex.LeftKnee,
            JointIndex.LeftAnkle, JointIndex.LeftAnkleRoll,
            JointIndex.RightHipPitch, JointIndex.RightHipRoll,
            JointIndex.RightHipYaw, JointIndex.RightKnee,
            JointIndex.RightAnkle, JointIndex.RightAnkleRoll
        ]
        
        print("🚀 初始化完成，等待状态同步...")
        time.sleep(0.5)

    
    def _state_callback(self, msg: LowState_):
        """状态回调"""
        self.current_state = msg
    
    def _send_cmd(self, msg: LowCmd_):
        """发送指令并计算CRC"""
        msg.crc = self.crc.Crc(msg)
        self.publisher.Write(msg)  # 大写 Write
    
    def _create_empty_cmd(self):
        # 【修正】使用默认初始化函数，自动填充所有字段
        return unitree_hg_msg_dds__LowCmd_()
    
    def get_current_positions(self):
        """获取所有关节当前位置"""
        positions = {}
        if self.current_state is None:
            return positions
            
        for i in range(len(self.current_state.motor_state)):
            positions[i] = self.current_state.motor_state[i].q
        return positions
    
    def smooth_interpolate(self, current, target, max_delta):
        """平滑插值"""
        delta = target - current
        if abs(delta) > max_delta:
            delta = max_delta if delta > 0 else -max_delta
        return current + delta
    
    def set_weight(self, weight, duration=2.0):
        """设置控制权重"""
        steps = int(duration / CTRL_DT)
        delta = (weight - self.weight) / steps
        
        msg = self._create_empty_cmd()
        
        for i in range(steps):
            self.weight += delta
            msg.motor_cmd[JointIndex.NotUsedJoint].q = self.weight
            
            # 保持所有关节当前状态
            current_pos = self.get_current_positions()
            for joint_idx in range(29):
                if joint_idx in current_pos:
                    msg.motor_cmd[joint_idx].q = current_pos[joint_idx]
                msg.motor_cmd[joint_idx].dq = 0.0
                msg.motor_cmd[joint_idx].kp = self.kp_lower
                msg.motor_cmd[joint_idx].kd = self.kd_lower
                msg.motor_cmd[joint_idx].tau = 0.0
            
            self._send_cmd(msg)
            time.sleep(CTRL_DT)
        
        self.weight = weight
    
    def move_to_pose(self, target_poses, duration=3.0):
        """
        移动到目标姿态
        target_poses: dict {JointIndex: target_angle}
        """
        steps = int(duration / CTRL_DT)
        max_delta = self.max_joint_velocity * CTRL_DT
        
        # 记录起始位置
        start_poses = self.get_current_positions()
        
        # 当前插值位置
        current_poses = {k: v for k, v in start_poses.items()}
        
        msg = self._create_empty_cmd()
        
        for step in range(steps):
            # 更新目标关节
            for joint_idx, target_angle in target_poses.items():
                current_poses[joint_idx] = self.smooth_interpolate(
                    current_poses[joint_idx], target_angle, max_delta
                )
            
            # 填充所有关节指令
            for joint_idx in range(29):
                # 设置目标位置
                if joint_idx in current_poses:
                    msg.motor_cmd[joint_idx].q = current_poses[joint_idx]
                
                # 设置刚度和阻尼
                if joint_idx in self.leg_joints:
                    # 腿部：高刚度保持站立
                    msg.motor_cmd[joint_idx].kp = self.kp_lower
                    msg.motor_cmd[joint_idx].kd = self.kd_lower
                elif joint_idx in self.arm_joints:
                    # 手臂：正常刚度
                    msg.motor_cmd[joint_idx].kp = self.kp_upper
                    msg.motor_cmd[joint_idx].kd = self.kd_upper
                else:
                    # 其他关节
                    msg.motor_cmd[joint_idx].kp = self.kp_upper
                    msg.motor_cmd[joint_idx].kd = self.kd_upper
                
                msg.motor_cmd[joint_idx].dq = 0.0
                msg.motor_cmd[joint_idx].tau = 0.0
            
            # 权重
            msg.motor_cmd[JointIndex.NotUsedJoint].q = self.weight
            
            self._send_cmd(msg)
            time.sleep(CTRL_DT)
                # 【新增】打印最终误差

        print("\n--- 姿态到达检查 ---")
        final_positions = self.get_current_positions()
        for joint_idx, target_angle in target_poses.items():
            if joint_idx in final_positions:
                error = abs(final_positions[joint_idx] - target_angle)
                if error > 0.1:
                    print(f"  关节{joint_idx}: 目标={target_angle:.2f}, 实际={final_positions[joint_idx]:.2f}, 误差={error:.2f}")
        print("-------------------\n")

        
        return current_poses
    
    def run_action_sequence(self):
        """执行动作序列"""
        print("\n" + "=" * 50)
        print("开始动作序列")
        print("=" * 50)
        
        # 等待状态数据到达
        print("等待机器人状态数据...")
        timeout = 5.0
        start_time = time.time()
        while self.current_state is None:
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                print("[ERROR] 未收到状态数据！请检查网络连接或 DDS 话题。")
                return

        # 读取当前位置
        current = self.get_current_positions()
        print("当前关节位置已读取")
        
        # 阶段 1: 获取控制权 (权重 0 -> 1)
        print("\n[1/5] 获取控制权...")
        self.set_weight(1.0, duration=2.0)
        
        # 阶段 2: 初始化姿态 (归零)
        print("\n[2/5] 初始化姿态...")
        init_poses = {idx: 0.0 for idx in self.arm_joints}
        init_poses.update({idx: current[idx] for idx in self.leg_joints})
        self.move_to_pose(init_poses, duration=2.0)
        
        # 阶段 3: 抬起手臂
        print("\n[3/5] 抬起手臂...")
        half_pi = math.pi / 2
        raise_poses = {
            JointIndex.LeftShoulderPitch: 0.0,
            JointIndex.LeftShoulderRoll: half_pi,
            JointIndex.LeftShoulderYaw: 0.0,
            JointIndex.LeftElbowPitch: half_pi,
            JointIndex.LeftElbowRoll: 0.0,
            JointIndex.RightShoulderPitch: 0.0,
            JointIndex.RightShoulderRoll: -half_pi,
            JointIndex.RightShoulderYaw: 0.0,
            JointIndex.RightElbowPitch: half_pi,
            JointIndex.RightElbowRoll: 0.0,
            JointIndex.WaistYaw: 0.0,
            JointIndex.WaistRoll: 0.0,
            JointIndex.WaistPitch: 0.0,
        }
        # 保持腿部位置
        raise_poses.update({idx: current[idx] for idx in self.leg_joints})
        self.move_to_pose(raise_poses, duration=3.0)
        
        # 阶段 4: 比耶动作
        print("\n[4/5] 比耶动作...")
        peace_poses = {
            JointIndex.LeftShoulderPitch: 0.0,
            JointIndex.LeftShoulderRoll: -0.50,
            JointIndex.LeftShoulderYaw: 0.0,
            JointIndex.LeftElbowPitch: half_pi,
            JointIndex.LeftElbowRoll: 0.0,
            # 右臂比耶
            JointIndex.RightShoulderPitch: -half_pi,
            JointIndex.RightShoulderRoll: -half_pi,
            JointIndex.RightShoulderYaw: 0.0,
            JointIndex.RightElbowPitch: -half_pi,    # 肘部微弯
            JointIndex.RightElbowRoll: 1.0,    # 手腕旋转
            JointIndex.WaistYaw: 0.0,
            JointIndex.WaistRoll: 0.0,
            JointIndex.WaistPitch: 0.0,
        }
        peace_poses.update({idx: current[idx] for idx in self.leg_joints})
        self.move_to_pose(peace_poses, duration=2.0)


        # 保持动作 2 秒
        print("保持动作...")
        time.sleep(2.0)
        
        # 阶段 5: 放下手臂
        print("\n[5/5] 放下手臂...")
        final_poses = {idx: 0.0 for idx in self.arm_joints}
        final_poses.update({idx: current[idx] for idx in self.leg_joints})
        self.move_to_pose(final_poses, duration=3.0)
        
        # 阶段 6: 释放控制权
        print("\n释放控制权...")
        self.set_weight(0.0, duration=2.0)
        
        print("\n动作序列完成！")


# ==================== 主函数 ====================
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <network_interface>")
        print("Example: python3 g1_action.py eth0")
        sys.exit(-1)
    
    controller = G1ActionController(sys.argv[1])
    
    print("\n按 Enter 开始执行动作...")
    input()
    
    controller.run_action_sequence()

if __name__ == "__main__":
    main()
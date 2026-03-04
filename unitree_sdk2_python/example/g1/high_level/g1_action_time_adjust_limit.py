#!/usr/bin/env python3
"""
G1 动作控制器 (Python版) - 带软限位和时间自动估计
- 功能：在机器人站立时控制上半身做动作（如比耶、抬手）
- 安全：自动锁定腿部关节，保持站立姿态
- 限位：软限位保护，防止发送超限指令
- 时间：自动计算动作所需时间，无需手动指定
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
        self.time_buffer = 1.0   # 时间缓冲（秒）
        
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
        
        # 【关节软限位配置】(单位: 弧度)
        # 基于 G1 官方 URDF 和实际测试
        self.joint_limits = {
            # 左腿
            JointIndex.LeftHipPitch: (-2.35, 2.35),
            JointIndex.LeftHipRoll: (-0.5, 0.5),
            JointIndex.LeftHipYaw: (-0.5, 0.5),
            JointIndex.LeftKnee: (-2.5, 2.5),
            JointIndex.LeftAnkle: (-0.8, 0.8),
            JointIndex.LeftAnkleRoll: (-0.5, 0.5),
            # 右腿
            JointIndex.RightHipPitch: (-2.35, 2.35),
            JointIndex.RightHipRoll: (-0.5, 0.5),
            JointIndex.RightHipYaw: (-0.5, 0.5),
            JointIndex.RightKnee: (-2.5, 2.5),
            JointIndex.RightAnkle: (-0.8, 0.8),
            JointIndex.RightAnkleRoll: (-0.5, 0.5),
            # 腰部
            JointIndex.WaistYaw: (-1.0, 1.0),
            JointIndex.WaistRoll: (-0.5, 0.5),
            JointIndex.WaistPitch: (-0.5, 0.5),
            # 左臂
            JointIndex.LeftShoulderPitch: (-3.14, 2.09),
            JointIndex.LeftShoulderRoll: (-0.5, 3.14),
            JointIndex.LeftShoulderYaw: (-2.09, 2.09),
            JointIndex.LeftElbowPitch: (-2.5, 0.0),
            JointIndex.LeftElbowRoll: (-1.57, 1.57),
            # 右臂
            JointIndex.RightShoulderPitch: (-3.14, 2.09),
            JointIndex.RightShoulderRoll: (-3.14, 0.5),
            JointIndex.RightShoulderYaw: (-2.09, 2.09),
            JointIndex.RightElbowPitch: (-2.5, 0.0),
            JointIndex.RightElbowRoll: (-1.57, 1.57),
        }
        
        print("🚀 初始化完成，等待状态同步...")
        time.sleep(0.5)

    
    def _state_callback(self, msg: LowState_):
        """状态回调"""
        self.current_state = msg
    
    def _send_cmd(self, msg: LowCmd_):
        """发送指令并计算CRC"""
        msg.crc = self.crc.Crc(msg)
        self.publisher.Write(msg)
    
    def _create_empty_cmd(self):
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
    
    # ==================== 软限位功能 ====================
    def clamp_to_limits(self, joint_idx, angle):
        """将角度限制在安全范围内"""
        if joint_idx in self.joint_limits:
            lower, upper = self.joint_limits[joint_idx]
            if angle < lower:
                return lower
            elif angle > upper:
                return upper
        return angle
    
    def check_and_clamp_poses(self, target_poses, verbose=True):
        """检查并修正所有目标角度"""
        clamped_poses = {}
        clamped_joints = []
        
        for joint_idx, target_angle in target_poses.items():
            original = target_angle
            clamped_angle = self.clamp_to_limits(joint_idx, target_angle)
            if abs(clamped_angle - original) > 0.01:
                clamped_joints.append((joint_idx, original, clamped_angle))
            clamped_poses[joint_idx] = clamped_angle
        
        if verbose and clamped_joints:
            print("  ⚠️ 关节限位修正:")
            for joint_idx, original, clamped in clamped_joints:
                print(f"    关节{joint_idx}: {original:.2f} → {clamped:.2f}")
        
        return clamped_poses
    
    # ==================== 时间自动估计功能 ====================
    def calculate_duration(self, current_poses, target_poses, min_duration=2.0):
        """
        根据关节位移自动计算所需时间
        
        参数:
            current_poses: 当前关节位置字典
            target_poses: 目标关节位置字典（应该是已经限位修正过的）
            min_duration: 最小持续时间
        
        返回:
            预估所需时间（秒）
        """
        max_travel = 0.0
        max_joint = None
        
        for joint_idx, target_angle in target_poses.items():
            if joint_idx in current_poses:
                travel = abs(target_angle - current_poses[joint_idx])
                if travel > max_travel:
                    max_travel = travel
                    max_joint = joint_idx
        
        if max_travel < 0.01:
            return min_duration
        
        # 时间 = 行程 / 速度 + 缓冲
        needed_time = max_travel / self.max_joint_velocity + self.time_buffer
        duration = max(needed_time, min_duration)
        
        print(f"  ⏱️ 时间估计: 最大行程 {max_travel:.2f}rad (关节{max_joint}), 需要 {duration:.1f}秒")
        
        return duration
    
    # ==================== 核心控制方法 ====================
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
    
    def move_to_pose(self, target_poses, duration=None, min_duration=2.0):
        """
        移动到目标姿态（支持自动时间估计）
        
        参数:
            target_poses: dict {JointIndex: target_angle}
            duration: 持续时间，如果为 None 则自动计算
            min_duration: 最小持续时间（自动计算时的下限）
        """
        # 步骤1: 检查并修正目标角度（软限位）
        clamped_poses = self.check_and_clamp_poses(target_poses)
        
        # 步骤2: 获取当前位置
        start_poses = self.get_current_positions()
        current_poses = {k: v for k, v in start_poses.items()}
        
        # 步骤3: 计算时间（如果未指定）
        if duration is None:
            duration = self.calculate_duration(start_poses, clamped_poses, min_duration)
        else:
            print(f"  ⏱️ 使用指定时间: {duration:.1f}秒")
        
        # 步骤4: 执行插值运动
        steps = int(duration / CTRL_DT)
        max_delta = self.max_joint_velocity * CTRL_DT
        
        msg = self._create_empty_cmd()
        
        for step in range(steps):
            # 更新目标关节
            for joint_idx, target_angle in clamped_poses.items():
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
                    msg.motor_cmd[joint_idx].kp = self.kp_lower
                    msg.motor_cmd[joint_idx].kd = self.kd_lower
                elif joint_idx in self.arm_joints:
                    msg.motor_cmd[joint_idx].kp = self.kp_upper
                    msg.motor_cmd[joint_idx].kd = self.kd_upper
                else:
                    msg.motor_cmd[joint_idx].kp = self.kp_upper
                    msg.motor_cmd[joint_idx].kd = self.kd_upper
                
                msg.motor_cmd[joint_idx].dq = 0.0
                msg.motor_cmd[joint_idx].tau = 0.0
            
            # 权重
            msg.motor_cmd[JointIndex.NotUsedJoint].q = self.weight
            
            self._send_cmd(msg)
            time.sleep(CTRL_DT)
        
        # 步骤5: 打印最终误差
        self._print_pose_error(clamped_poses)
        
        return current_poses
    
    def _print_pose_error(self, target_poses):
        """打印姿态到达误差"""
        final_positions = self.get_current_positions()
        errors = []
        
        for joint_idx, target_angle in target_poses.items():
            if joint_idx in final_positions:
                error = abs(final_positions[joint_idx] - target_angle)
                if error > 0.1:
                    errors.append((joint_idx, target_angle, final_positions[joint_idx], error))
        
        if errors:
            print("\n  --- 姿态到达检查 ---")
            for joint_idx, target, actual, error in errors:
                print(f"    关节{joint_idx}: 目标={target:.2f}, 实际={actual:.2f}, 误差={error:.2f}rad")
            print("  -------------------\n")
    
    def hold_pose(self, duration=2.0):
        """保持当前姿态一段时间（持续发送命令防止下垂）"""
        steps = int(duration / CTRL_DT)
        msg = self._create_empty_cmd()
        
        for _ in range(steps):
            current_pos = self.get_current_positions()
            
            for joint_idx in range(29):
                if joint_idx in current_pos:
                    msg.motor_cmd[joint_idx].q = current_pos[joint_idx]
                
                if joint_idx in self.leg_joints:
                    msg.motor_cmd[joint_idx].kp = self.kp_lower
                    msg.motor_cmd[joint_idx].kd = self.kd_lower
                elif joint_idx in self.arm_joints:
                    msg.motor_cmd[joint_idx].kp = self.kp_upper
                    msg.motor_cmd[joint_idx].kd = self.kd_upper
                else:
                    msg.motor_cmd[joint_idx].kp = self.kp_upper
                    msg.motor_cmd[joint_idx].kd = self.kd_upper
                
                msg.motor_cmd[joint_idx].dq = 0.0
                msg.motor_cmd[joint_idx].tau = 0.0
            
            msg.motor_cmd[JointIndex.NotUsedJoint].q = self.weight
            
            self._send_cmd(msg)
            time.sleep(CTRL_DT)
    
    # ==================== 动作序列 ====================
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
        
        # 阶段 1: 获取控制权
        print("\n[1/5] 获取控制权...")
        self.set_weight(1.0, duration=2.0)
        
        # 阶段 2: 初始化姿态 (归零)
        print("\n[2/5] 初始化姿态...")
        init_poses = {idx: 0.0 for idx in self.arm_joints}
        init_poses.update({idx: current[idx] for idx in self.leg_joints})
        self.move_to_pose(init_poses)  # 自动计算时间
        
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
        raise_poses.update({idx: current[idx] for idx in self.leg_joints})
        self.move_to_pose(raise_poses)  # 自动计算时间
        
        # 阶段 4: 比耶动作
        print("\n[4/5] 比耶动作...")
        peace_poses = {
            JointIndex.LeftShoulderPitch: 0.0,
            JointIndex.LeftShoulderRoll: -0.50,
            JointIndex.LeftShoulderYaw: 0.0,
            JointIndex.LeftElbowPitch: 0.0,
            JointIndex.LeftElbowRoll: 0.0,
            # 右臂比耶
            JointIndex.RightShoulderPitch: -half_pi * 2,
            JointIndex.RightShoulderRoll: -half_pi,
            JointIndex.RightShoulderYaw: 0.0,
            JointIndex.RightElbowPitch: -1.5,
            JointIndex.RightElbowRoll: 1.0,
            JointIndex.WaistYaw: 0.0,
            JointIndex.WaistRoll: 0.0,
            JointIndex.WaistPitch: 0.0,
        }
        peace_poses.update({idx: current[idx] for idx in self.leg_joints})
        self.move_to_pose(peace_poses)  # 自动计算时间
        
        # 保持动作
        print("保持动作...")
        self.hold_pose(2.0)

        # 阶段 5: 放下手臂
        print("\n[5/5] 放下手臂...")
        final_poses = {idx: 0.0 for idx in self.arm_joints}
        final_poses.update({idx: current[idx] for idx in self.leg_joints})
        self.move_to_pose(final_poses)  # 自动计算时间
        
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

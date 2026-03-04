# Unitree G1 导航与语音助手项目 (WK)

本项目是在宇树 G1 人形机器人平台上，实现 **3D激光SLAM建图、自主导航、语音交互与运动控制** 的集成解决方案。

项目核心功能：
- 基于 FAST-LIO2 的3D激光里程计与建图
- 离线点云地图定位与重定位
- 2D栅格地图导航（支持 TEB 局部规划器）
- 集成语音助手，支持语音指令控制机器人运动
- 宇树 G1 SDK 运动控制示例（含 MPC 控制）

---

## 目录

- [项目结构](#项目结构)
WK/
├── G1Nav2D/ # ROS 导航核心工作空间
│ ├── src/
│ │ ├── fastlio2/ # FAST-LIO2 SLAM 核心包
│ │ ├── livox_ros_driver2/ # Livox 激光雷达驱动
│ │ ├── movebase/ # 导航栈配置（TEB规划器等）
│ │ ├── pointcloud_to_laserscan/ # 点云转激光扫描
│ │ ├── ros_map_edit/ # RViz 地图编辑插件
│ │ ├── tool/ # 点云处理、路径记录等工具
│ │ └── velocity_smoother_ema/ # 速度平滑器
│ └── …
├── PythonProject/ # 上层应用：语音助手与导航脚本
│ ├── py-xiaozhi-main/ # 语音助手（小智）集成
│ ├── daohang/ # 特定场景导航脚本（电梯、卫生间）
│ └── point_nav/ # 点对点导航示例
└── unitree_sdk2_python/ # 宇树机器人 Python SDK 
└── example/g1/high_level/ # G1 高级控制示例（MPC等 还有自定义导航点发布和导航讲解

- [环境依赖](#环境依赖)

---

## 环境依赖

### 硬件环境
- 宇树 G1 人形机器人
- Livox MID360 激光雷达
- 计算平台（机器人自带 ARM 板或外部 PC）

### 软件环境
- Ubuntu 20.04
- ROS Noetic
- Python 3.8+
- 依赖库：Eigen3, PCL, Open3D, Livox-SDK2 等

---

## 快速开始

### 1. 克隆仓库
```bash

git clone https://github.com/Dailywatero/WK.git

cd WK
```

### 2. 安装 Livox SDK2
```bash


sudo apt install cmake
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
mkdir build && cd build
cmake … && make -j
sudo make install
```

### 3. 修改雷达 IP 及地图路径

1. 修改本机与雷达 IP：
```bash
cd WK/G1Nav2D/src/livox_ros_driver2/config
gedit MID360_config.json
```
根据你的网络环境修改 “lidar_configs” 里的雷达 IP 和 “host_net_info” 里的本机 IP
雷达ip一般不用修改，192.168.123.120
用ifconfig查自己的ip，要在同一网段下


2. 修改地图保存路径：
```bash
cd WK/G1Nav2D/src/fastlio2/src
gedit map_builder_node.cpp
```
将文件中地图保存路径（如 /home/water/map/）改为你自己的路径


### 4. 编译 ROS 工作空间
```bash
cd WK/G1Nav2D
```
建议先单独编译 livox_ros_driver2 和 fastlio2，避免消息头文件找不到
```bash
catkin_make -DROS_EDITION=ROS1 --pkg livox_ros_driver2
catkin_make -DROS_EDITION=ROS1 --pkg fastlio2

#整体编译

catkin_make
source devel/setup.bash
```

### 5. 安装导航相关依赖
```bash
sudo apt install ros-noetic-teb-local-planner
ros-noetic-global-planner
ros-noetic-costmap-server
```

---

## 使用说明

### 1. 建图与保存地图

**建图：**
```bash
cd WK/G1Nav2D
source devel/setup.bash
roslaunch fastlio2 mapping.launch
```

**保存地图：**

新开终端：
```bash
cd WK/G1Nav2D
source devel/setup.bash
rosrun map_server map_saver map:=/projected_map -f /home/你的用户名/map/mymap
```

### 2. 编辑地图
```bash
source devel/setup.bash
roslaunch ros_map_edit map_edit.launch
```

下载photogimp.desktop
在里面编辑pgm地图文件

### 3. 启动导航

1. 修改地图加载路径：
```bash
cd WK/G1Nav2D/src/fastlio2/launch
gedit gridmap_load.launch
```
修改 2dmap_file 参数为你保存的地图路径，例如：
<arg name="2dmap_file" default="/home/你的用户名/map/mymap.yaml" />

2. 启动导航：
```bash
cd WK/G1Nav2D
source devel/setup.bash
roslaunch fastlio2 navigation.launch
```
3. 初始重定位（如有需要）：
```bash
rosservice call /slam_reloc
“{pcd_path: ‘/home/你的用户名/WK/G1Nav2D/src/fastlio2/PCD/map.pcd’,
x: 0.0, y: 0.0, z: 0.0, roll: 0.0, pitch: 0.0, yaw: 0.0}”
```

在 RViz 中确认点云与地图是否配准良好。

### 4. 运动控制（G1）

参考宇树官方文档安装 `unitree_sdk2_python`：
```bash
cd WK/unitree_sdk2_python/example/g1/high_level
python3 g1_control_mpc.py eth0 # 或其他 g1_control_*.py
```

### 5. 语音交互导航（基于 py-xiaozhi）

详细配置请参考 `PythonProject/py-xiaozhi-main/README.md`，这里给出关键步骤。

**安装依赖：**
```bash
cd WK/PythonProject/py-xiaozhi-main
pip install -r requirements.txt
```

**语音导航到目标点（简易版）：**

1. 在 `PythonProject/point_nav/point2.py` 中修改目标点坐标（文件最底部）。
2. 全局搜索“电梯”，根据需要替换为“卧室”、“卫生间”等关键词，并修改对应坐标。

**语音导航到目标点（MCP 版）：**

1. 在 `PythonProject/py-xiaozhi-main/src/application.py` 中添加/修改关键词与对应的 MCP 服务。
2. 在 `PythonProject/py-xiaozhi-main/src/mcp/mcp_server.py` 中注册 MCP 服务。
3. 在 `PythonProject/py-xiaozhi-main/src/mcp/tools/daohang_dianti/tools.py` 中配置拉起的 Python 程序及解释器路径。
4. 在 `PythonProject/daohang/daohang-dianti.py` 中修改目标点坐标。

**启动语音程序：**
```bash
cd WK/PythonProject/py-xiaozhi-main
python3 main.py
```

> 实现语音交互导航需要同时启动：语音程序、运控程序、导航程序。

---

## 核心模块说明

### 1. G1Nav2D (导航核心)
- `fastlio2/`: 实现 3D SLAM 与定位。
- `movebase/`: 集成 ROS navigation stack，使用 TEB 局部规划器进行避障与路径规划。
- `ros_map_edit/`: 提供 RViz 插件，方便编辑和修正 2D 栅格地图。

### 2. PythonProject (上层应用)
- `py-xiaozhi-main/`: 一个功能完整的语音助手项目，集成了大语言模型、物联网控制等功能，通过语音指令控制机器人。
- `point_nav/`, `daohang/`: 封装好的导航任务脚本，实现“去电梯口”、“去卫生间”等特定场景的自主导航。

### 3. unitree_sdk2_python (运动控制)
提供对宇树 G1 机器人的底层和高级控制接口，本项目重点使用了其中的 MPC 控制示例来实现平稳的运动控制。

---

## 常见问题

**Q: 编译时出现 `CustomMsg.h` 或 `SlamReLoc.h` 找不到？**  
A: 需要先单独编译消息生成包，参考“快速开始”中的编译顺序。

**Q: 如何修改地图路径？**  
A: 在 `G1Nav2D/src/fastlio2/launch/gridmap_load.launch` 中修改 `2dmap_file` 参数。
---

## 致谢

- [FAST_LIO](https://github.com/hku-mars/FAST_LIO)
- [Livox-SDK2](https://github.com/Livox-SDK/Livox-SDK2) 及 `livox_ros_driver2`
- [ros_map_edit](https://github.com/你的参考链接)
- [py-xiaozhi](https://github.com/huangjunsen0406/py-xiaozhi)
- 宇树科技 Unitree
- 以及所有开源项目贡献者。

---

## License

本项目仅供学习交流，请遵守相关开源协议。
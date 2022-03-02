## 1.环境配置
本代码在PC端Ubuntu18.04和rv1126嵌入式板子中的Linux系统上均通过测试。

代码的编译和运行环境支持x86-64架构和ARM32架构。
##### 1.1程序依赖的第三方库

程序所依赖的第三方库需要使用者自行安装，编译
其中，运行在嵌入式端的程序需要借助交叉编译工具链（gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf）来进行编译

PC端和嵌入式端共同使用的库：
+ opencv 3.4.5版本带contrib
+ Eigen3
+ PCL 1.8（features, filters, segmentation, octree, search, sample_consensus, kdtree, common, ml）
+ Boost 1.65.0(嵌入式端用的1.75.0)
+ Sophus
+ Thirdparty/g2o
+ Thirdparty/DBow3

PC端独有的库：
+ PCL所有库文件(包含了可视化部分)
+ Pangolin（用于可视化）

嵌入式端独有的库：
+ RKNN
+ RGA
+ DRM

## 2.程序编译

CMakelists.txt文件中设置了两种不同的编译模式，分别对应于PC端和嵌入式端的编译环境
+ 若在PC端进行编译，system_frame设置为"Linux64"
+ 若在嵌入式端进行编译， system_frame设置为"ARM32"

除此之外，需要在CMakelists.txt文件中修改上述所有第三方库的搜索路径

备注：
+ 嵌入式端的程序需要在PC端编译完成后，将所有程序依赖的动态库文件上传至嵌入式端
+ 可利用"adb push 本地源文件路径 嵌入式端目标路径"来上传本地文件至嵌入式端；可利用"adb pull 嵌入式端文件路径 本地目标路径"来拉取嵌入式端文件至本地
+ PC端默认将可执行文件保存在YOUR_PATH/sweepSLAM/bin目录下，若路径不一致则需在运行时进行修改
+ 嵌入式端默认将可执行文件保存在/data/ttt，将所有配置文件保存在/data/ttt/initial，将库文件存放在两个文件夹，分别为/data/ttt/lib/dep_libs/和/pcl_libs/目录下，若路径不一致则需在运行时进行修改
+ config_on_rv1126文件夹为嵌入式端的所有配置文件，可一并上传

## 3.整体模块的运行

### 在运行之前，需要修改各模块yaml配置文件中的读取数据路径和运行结果保存路径，分别对应于data_path和save_path两个变量。
```
# PC端：
cd YOUR_PATH/sweepSLAM/bin
./sweeper_robot ../config/robot/sweeper_config.yaml

# 嵌入式端：
cd /data/ttt/ 
LD_LIBRARY_PATH=/data/ttt/lib/dep_libs/:/pcl_libs/ ./sweeper_robot ./initial/sweeper_config.yaml
```

### 各模块分别运行
### 3.1 slam
数据集：/10_200s/文件夹
SLAM模块可以选择逐帧运行（默认方式），也可以选择在已有数据集的基础上模拟实时运行的状态得到结果。
```
# PC端：
cd YOUR_PATH/sweepSLAM/bin
./stereo_odo_3 ../config/robot/robot_orb_stereo_new.yaml

# 嵌入式端：
cd /data/ttt/ 
LD_LIBRARY_PATH=/data/ttt/lib/dep_libs/:/pcl_libs/ ./stereo_odo_3 ./initial/robot_orb_stereo_new.yaml
```
##### 评估轨迹误差(在PC端运行)
```js
cd /YOUR_PROJECT_PATH/sweepSLAM/evaluation/liegroups && pip install . && cd ..
python3 eval_self.py --est_path /YOUR_SAVE_PATH/orb3_stereo_slam.txt --gt_path ../config/gt/vicon_10.txt --end_time 1601012055.929288
```
   + [evo](https://github.com/MichaelGrupp/evo.git)
        + numpy,scipy,matplotlib
       + 仅用于可视化,不用于计算精确度,可以不安装
       + 因为后期超过了vicon的范围,所以不能评估全部轨迹
       +  轨迹评估标准是当前误差比上以走路程,单位百分比：
       + trans error mean: xxx
       + trans error median: xxx
       + rot error mean: xxx
       + rot error median: xxx

### 3.2主方向识别

数据集：/10_200s/文件夹
```
# PC端：
cd YOUR_PATH/sweepSLAM/bin
./main_direction_detection ../config/robot/main_direction_detector.yaml

# 嵌入式端：
cd /data/ttt/ 
LD_LIBRARY_PATH=/data/ttt/lib/dep_libs/:/pcl_libs/ ./main_direction_detection ./initial/main_direction_detector.yaml
```

主方向识别只在初期运行,得到主方向之后就停止运行。主方向识别利用房间中和房间垂直的直线信息,要求运行期间机器人运动尽量平稳,可以有一些旋转运动.
运行后，结果保存在save_path下的main_direction_result.txt文件中。
各列表示：

(1) 用于主方向识别的50帧中第一帧的时间戳； 

(2) 房间主方向相对于相机的角度值； 

(3) 置信度； 

真实值通过vicon测量得到，保存在/YOUR_PROJECT_PATH/sweepSLAM/evaluation/main_direction_detection_result/目录下的main_direction_gt.txt文件中，可通过时间戳将检测结果和真实值对应和比较。

### 3.3 四类物体检测

数据集：/exp0702/文件夹

```
# PC端：
cd YOUR_PATH/sweepSLAM/bin
./object_detector ../config/robot/object_detector.yaml

# 嵌入式端：
cd /data/ttt/
LD_LIBRARY_PATH=/data/ttt/lib/dep_libs/:/pcl_libs/ ./object_detector ./ini
tial/object_detector.yaml
```
物体检测结果存放于save_path下的object_det_matlab_sgbm.txt， 一行表示一个识别到的物体，各列分别表示：

(1) 帧序号

(2) 该帧yolo检测到的物体数量

(3) 该物体类别序号

(4) 该物体世界坐标系下x、y、z坐标(m)

(5) 该物体宽w(m)；

(6) 该物体高h(m)。



#### 计算前四类物体精确度(PC端)
```js
python3 /YOUR_PROJECT_PATH/sweepSLAM/evaluation/object_det_eval/eval_two_dimension.py  /YOUR_PROJECT_PATH/sweepSLAM/evaluation/obj_boxes_result/ obj_boxes.txt
``` 

识别精确度输出如下,前四行分别输出四种物体各自的识别成功个数,总个数,识别精确度,最后一行输出总的识别精确度。
```
////////////////classification result//////////////////
Precision: 
Object       True Positive   Total Images(TP+FP)     Precision
Badminton:       xxx                xxx                 xxx
Wire:            xxx                xxx                 xxx
Mahjong:         xxx                xxx                 xxx
Dog Shit:        xxx                xxx                 xxx
Total Average:   xxx                xxx                 xxx

```

### 3.4 重定位
+ 注意：词典训练在PC端进行，重定位程序可在PC端ubuntu系统或嵌入式开发板上运行，可对重定位精度进行验证
+ 数据集/01/、/02/、/03/文件夹，以下所有指令中需要手动修改YOUR_PATH以及YOUR_PATH/dataset/（存放/01/、/02/、/03/文件夹的目录）
+ 利用一个序列进行词典训练(训练点线词典)，训练数据集：/03/文件夹
```
# PC端训练：
cd YOUR_PATH/to_YX_PC/executable
./feature_training YOUR_PATH/dataset/
```

+ 在当前目录生成点词典 robot_vocab_pt.yml.gz
+ 在当前目录生成线词典 robot_vocab_line.yml.gz


+ 利用另外两个序列进行重定位,用于测试成功率，测试数据集：/01/、/02/文件夹
```
# PC端：
cd YOUR_PATH/to_YX_PC/executable
./relocalization YOUR_PATH/dataset/

# 嵌入式端：
LD_LIBRARY_PATH=/data/ttt/lib/dep_libs/:/pcl_libs/  /data/ttt/eval  YOUR_PATH/dataset/
```
+ 在当前目录生成结果 result_dbow_pl.txt, 每行第一个是当前图片序号,后面是按相似程度排序的重定位图片序号结果
+ 这一步运行时间较长

+ 测试成功率
```
cd YOUR_PATH/pyeval(PC端与嵌入式端均使用该测试方法)
python generate_dataset.py --path YOUR_PATH/dataset/
```
+ 用vicon数据生成重定位真值,认为旋转小于15度,平移小于0.2m是在相同地点
+ 生成真值文件test_positive_15_0.2.txt
+ 请把之前生成的result_dbow_pl.txt复制到当前文件夹
```
python generate_dataset.py --path YOUR_PATH/dataset/ --test
```
+ 每一行,例如50 97.4854338188727,代表,对于数据集中的一张图片,我们的算法找出的前50个最相似的结果都是对的,则记该张图片成功,统计整个数据集上的成功率.
   + 成功率影响因素:
        + 可以适当调小树深度;
        + 训练词典时尽量挑选有代表性图片,例如关键帧,而不是所有图片

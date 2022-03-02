#python eval_self.py --est_path /home/qzj/ros/FrameTrajectory_TUM_Format.txt --gt_path /home/qzj/code/catkin_ws/src/args/SLAM/vicon_10.txt --end_time 1601012055.929288 --scaleAlign
export SLAM_PATH="/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3"
export SEQ="07"
python eval_self.py --est_path ${SLAM_PATH}/${SEQ}/FrameTrajectory_TUM_Format.txt --gt_path ${SLAM_PATH}/${SEQ}/vicon_${SEQ}.txt --scaleAlign

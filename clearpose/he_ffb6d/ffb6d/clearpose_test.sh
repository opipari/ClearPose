# model_sensor=primesense
# dataset_sensor=primesense

# checkpoint=train_log/ycb/checkpoints/FFB6D_$model_sensor.pth.tar
# echo "model_sensor: " $model_sensor ", dataset_sensor: " $dataset_sensor

# # --test --sensor_name --template --ratio
# python /home/huijie/research/progresslabeller/FFB6D/ffb6d/datasets/ycb/dataset_config/generate_list.py test $dataset_sensor test_scene* 0.3

# python -m torch.distributed.launch --nproc_per_node=1 train_ycb.py --gpu 0 -eval_net -checkpoint $checkpoint -test -test_pose

# for model_sensor in primesense D435 L515; do
# for dataset_sensor in primesense D435 L515; do
#     if [ "$dataset_sensor" != "primesense" ] || [ "$model_sensor" != "primesense" ]; then
#         checkpoint=train_log/ycb/checkpoints/FFB6D_$model_sensor.pth.tar
#         echo "model_sensor: " $model_sensor ", dataset_sensor: " $dataset_sensor

#         # --test --sensor_name --template --ratio
#         python /home/huijie/research/progresslabeller/FFB6D/ffb6d/datasets/ycb/dataset_config/generate_list.py test $dataset_sensor test_scene* 0.3

#         python -m torch.distributed.launch --nproc_per_node=1 train_ycb.py --gpu 0 -eval_net -checkpoint $checkpoint -test -test_pose
#     fi
# done
# done

## r/r
# for test_type in color covered non-planner occlusion standard wou; do
#     touch outr_r_$test_type.txt
#     checkpoint=/home/huijie/research/transparentposeestimation/ClearPose/experiments/he_ffb6d/train_log/clearpose/checkpoints/FFB6D_train_raw.pth.tar
#     echo "r/r for test: " $test_type

#     python -m torch.distributed.launch --nproc_per_node=1 train_clearpose_test.py --gpu 0 -eval_net -checkpoint $checkpoint -test -test_type $test_type -depth_type raw -test_pose < outr_r_$test_type.txt

# done

for test_type in wou; do
    touch outgt_idmasked_$test_type.txt
    checkpoint=/home/huijie/research/transparentposeestimation/ClearPose/experiments/he_ffb6d/train_log/clearpose/checkpoints/FFB6D_train_gt.pth.tar
    echo "gt/idmask for test: " $test_type

    python -m torch.distributed.launch --nproc_per_node=1 train_clearpose_test.py --gpu 0 -eval_net -checkpoint $checkpoint -test -test_type $test_type -depth_type GT -test_pose

done



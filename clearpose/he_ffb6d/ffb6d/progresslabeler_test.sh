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

for model_sensor in baseline; do
for dataset_sensor in primesense D435 L515; do
    if [ "$dataset_sensor" != "primesense" ] || [ "$model_sensor" != "primesense" ]; then
        checkpoint=train_log/ycb/checkpoints/FFB6D_$model_sensor.pth.tar
        echo "model_sensor: " $model_sensor ", dataset_sensor: " $dataset_sensor

        # --test --sensor_name --template --ratio
        python /home/huijie/research/progresslabeller/FFB6D/ffb6d/datasets/ycb/dataset_config/generate_list.py test $dataset_sensor test_scene* 0.3

        python -m torch.distributed.launch --nproc_per_node=1 train_ycb.py --gpu 0 -eval_net -checkpoint $checkpoint -test -test_pose
    fi
done
done



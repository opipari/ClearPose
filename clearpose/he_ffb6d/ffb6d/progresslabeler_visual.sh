sensor=L515
checkpoint=train_log/ycb/checkpoints/FFB6D_fine_tune_$sensor.pth.tar

# --test --sensor_name --template
python datasets/ycb/dataset_config/generate_list.py test $sensor train_scene11  0.01

python demo.py -checkpoint $checkpoint -dataset ycb
# echo $tst_mdl 
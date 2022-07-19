
for test_type in wou; do
    touch outgt_idmasked_$test_type.txt
    checkpoint=/home/huijie/research/transparentposeestimation/ClearPose/experiments/he_ffb6d/train_log/clearpose/checkpoints/FFB6D_train_gt.pth.tar
    echo "gt/idmask for test: " $test_type

    python -m torch.distributed.launch --nproc_per_node=1 train_clearpose_test.py --gpu 0 -eval_net -checkpoint $checkpoint -test -test_type $test_type -depth_type GT -test_pose

done



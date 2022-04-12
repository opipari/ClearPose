for test_type in standardvideo occlusionvideo non-plannervideo coveredvideo colorvideo wouvideo; do
    echo "gt/tcgmask: " $test_type
    python -m datasets.clearpose.clearpose_dataset_gt -test_type $test_type -depth_type GT
done




#python3 tools/train_net.py --config-file configs/free_anchor_R-101-FPN_1x_DR_4lesions.yaml --skip-test 0 opts
export NGPUS=2
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/free_anchor_R-101-FPN_1x_DR_4lesions.yaml"
# --skip-test "false" 
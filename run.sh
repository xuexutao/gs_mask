python generate_multiview_sam_masks.py --source_path /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/data/room/ --images_subdir images_4 --num_workers 16

python assign_semantic_labels.py --source_path /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/data/room/ --categories chair,table,cut --mask_subdir images_4/masks_sam --output_labels labels.json

python train.py -s/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room -m output/test_0329 --semantic_labels /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room/images_4/masks_sam/labels.json --mask_dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room/images_4/masks_sam


#!/bin/bash

root='results/'
out_root='results/metrics'

celeba_array=('GeoMAR_celeba_test_144' 'GeoMAR_celeba_test_267')

other_array=('GeoMAR_lfw' 'GeoMAR_webphoto' 'GeoMAR_wider')

for test_name in "${celeba_array[@]}"
do
    test_image=$test_name'/restored_faces'
    # test_image=$test_name
    out_name=$test_name
    # 0: the name of image does not include 00 and Codeformer
    # 1: otherwise
    need_post=1
    # need_post=0

    CelebAHQ_GT='./datasets/celeba_512_validation'

    # FID
    python -u scripts/metrics/cal_fid.py \
    $root'/'$test_image \
    --fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
    --save_name $out_root'/'$out_name'_fid.txt' \

    python scripts/metrics/cal_niqe.py \
    $root'/'$test_image \
    --save_name $out_root'/'$out_name'_niqe.txt' \

    python scripts/metrics/cal_maniqa.py \
    $root'/'$test_image \
    --save_name $out_root'/'$out_name'_maniqa.txt' \


    if [ -d $CelebAHQ_GT ]
    then
        # PSRN SSIM LPIPS
        python -u scripts/metrics/cal_psnr_ssim.py \
        $root'/'$test_image \
        --gt_folder $CelebAHQ_GT \
        --save_name $out_root'/'$out_name'_psnr_ssim_lpips.txt' \
        --need_post $need_post \

    else
        echo 'The path of GT does not exist'
    fi
done

for test_name in "${other_array[@]}"
do
    test_image=$test_name'/restored_faces'
    # test_image=$test_name
    out_name=$test_name

    # FID
    python -u scripts/metrics/cal_fid.py \
    $root'/'$test_image \
    --fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
    --save_name $out_root'/'$out_name'_fid.txt' \

    python scripts/metrics/cal_niqe.py \
    $root'/'$test_image \
    --save_name $out_root'/'$out_name'_niqe.txt' \

    python scripts/metrics/cal_maniqa.py \
    $root'/'$test_image \
    --save_name $out_root'/'$out_name'_maniqa.txt' \

done
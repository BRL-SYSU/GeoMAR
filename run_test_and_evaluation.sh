#!/bin/bash

root='results'
out_root='results/metrics'

if [ ! -d $root ];then
    mkdir -p $root
fi

if [ ! -d $out_root ];then
    mkdir -p $out_root
fi


dataset_name_array=('test')


dataset_location_array=('celeba_test_144')


checkpoint='./experiments/GeoMAR_model.ckpt'
config='./configs/GeoMAR.yaml'
output_name='GeoMAR'
GPU='2'

# echo ${0}
echo ${checkpoint}
# echo ${config}
echo ${output_name}
echo $GPU
echo $dataset_name_array


outdir=$root'/'$output_name'_'${dataset_name_array[0]}
align_test_path='./datasets/'${dataset_location_array[0]}



CUDA_VISIBLE_DEVICES=$GPU python -u scripts/test.py \
--outdir $outdir \
-r $checkpoint \
-c $config \
--test_path $align_test_path \
--aligned



outdir=$output_name'_'${dataset_name_array[0]}

test_image=$outdir'/restored_faces'

out_name=$outdir


need_post=1


CelebAHQ_GT='/data/celeba_512_validation'

# FID
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/metrics/cal_fid.py \
$root'/'$test_image \
--fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
--save_name $out_root'/'$out_name'_fid.txt' \

CUDA_VISIBLE_DEVICES=$GPU python scripts/metrics/cal_niqe.py \
$root'/'$test_image \
--save_name $out_root'/'$out_name'_niqe.txt' \


CUDA_VISIBLE_DEVICES=$GPU python scripts/metrics/cal_maniqa.py \
$root'/'$test_image \
--save_name $out_root'/'$out_name'_maniqa.txt' \

if [ -d $CelebAHQ_GT ]
then
# PSRN SSIM LPIPS
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/metrics/cal_psnr_ssim.py \
$root'/'$test_image \
--gt_folder $CelebAHQ_GT \
--save_name $out_root'/'$out_name'_psnr_ssim_lpips.txt' \
--need_post $need_post \

else
    echo 'The path of GT does not exist'
fi
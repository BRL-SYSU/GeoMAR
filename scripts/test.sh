exp_name='GeoMAR'


root_path='experiments'
out_root_path='results'

tag='celeba_test_144'
align_test_path="./datasets/celeba_test_144"
eval_text_features_dir="./datasets/text_feature/celeba144"
outdir=$out_root_path'/'$exp_name'_'$tag

if [ ! -d $outdir ];then
    mkdir $outdir
fi

python -u scripts/test.py \
--outdir $outdir \
-r './experiments/GeoMAR_model.ckpt' \
-c 'configs/GeoMAR.yaml' \
--test_path $align_test_path \
--aligned \
model.params.eval_text_features_dir="$eval_text_features_dir"

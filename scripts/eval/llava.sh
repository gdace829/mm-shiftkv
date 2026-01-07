ratios=(0.1)
# methods=("sparsemm" "fullkv" "adakv" "snapkv" "pyramidkv" "mask" "mask_random")sparsemm_query
methods=("shiftkv")
budgets=(64)
mask_ratio=0.1 # only used for "mask" / "mask_random" flickr30k refcoco_seg_val ocrbench docvqa  video_dc499

for budget in ${budgets[@]}; do
    for ratio in ${ratios[@]}; do
        for method in ${methods[@]}; do
    
            export METHOD=${method}
            export BUDGET=${budget}
            export RATIO=${ratio}
            export MASK_RATIO=${mask_ratio}

            mkdir -p ./ocrbench_results/llama_results/

            export CUDA_VISIBLE_DEVICES=0,1,2,3
            python3 -u -m accelerate.commands.launch \
                --num_processes=4 \
                --main_process_port 54323\
                -m lmms_eval \
                --model llava \
                --model_args pretrained=/data/model/models--liuhaotian--llava-v1.6-vicuna-7b/snapshots/deae57a8c0ccb0da4c2661cc1891cc9d06503d11,conv_template=vicuna_v1 \
                --tasks ocrbench \
                --batch_size 1 \
                --log_samples \
                --log_samples_suffix llava_v1.6_mix \
                --output_path ./logs/ \
                --gen_kwargs temperature=0 \
                --verbosity=DEBUG 2>&1 | tee ./ocrbench_results/llama_results/ocrbench_${method}_${budget}_${ratio}.log
        done
    done
done

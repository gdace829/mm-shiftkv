ratios=(0.1)
methods=("sparsemm" "fullkv" "adakv" "snapkv" "pyramidkv" "mask" "mask_random")
budgets=(64 128 256 512 1024 2048)
mask_ratio=0.1 # only used for "mask" / "mask_random"

for budget in ${budgets[@]}; do
    for ratio in ${ratios[@]}; do
        for method in ${methods[@]}; do
            
            export METHOD=${method}
            export BUDGET=${budget}
            export RATIO=${ratio}
            export MASK_RATIO=${mask_ratio}

            mkdir -p ./ocrbench_results/mistral_results/

            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            python3 -m accelerate.commands.launch \
                --num_processes=8 \
                --main_process_port 54321\
                -m lmms_eval \
                --model llava \
                --model_args pretrained=/path/to/models/llava-mistral-v1.6,conv_template=mistral_instruct \
                --tasks ocrbench \
                --batch_size 1 \
                --log_samples \
                --log_samples_suffix llava_v1.6_mix \
                --output_path ./logs/ \
                --gen_kwargs temperature=0 \
                --verbosity=DEBUG 2>&1 | tee ./ocrbench_results/mistral_results/ocrbench_${method}_${budget}_${ratio}.log
        done
    done
done

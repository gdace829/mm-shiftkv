ratios=(0.1)
methods=("shiftkv") # "keydiff" "streamingllm" "expectedAttention" "sparsemm_query" "snapkv"
budgets=(64) # 128 256 512
tasks=("ocrbench") # "textvqa" "ocrbench" "docvqa" "chartqa" "textcaps" "mmmu_pro" "pope" "ok_vqa" "ST-VQA" "flicker30k"

mask_ratio=0.1

for task in "${tasks[@]}"; do
    for budget in "${budgets[@]}"; do
        for ratio in "${ratios[@]}"; do
            for method in "${methods[@]}"; do

                export METHOD=${method}
                export BUDGET=${budget}
                export RATIO=${ratio}
                export MASK_RATIO=${mask_ratio}

                mkdir -p ./results/${task}/llama_resultsfull/

                export CUDA_VISIBLE_DEVICES=0,1,2,3
                python3 -u -m accelerate.commands.launch \
                    --num_processes=4 \
                    --main_process_port 54323 \
                    -m lmms_eval \
                    --model qwen2_5_vl \
                    --model_args pretrained="Qwen/Qwen2.5-VL-7B-Instruct,use_flash_attention_2=True" \
                    --tasks ${task} \
                    --batch_size 1 \
                    --log_samples \
                    --log_samples_suffix llava_v1.6_mix \
                    --output_path ./logs/ \
                    --gen_kwargs temperature=0 \
                    --verbosity=DEBUG 2>&1 | tee \
                    ./results/${task}/llama_resultsfull/${task}_${method}_${budget}_${ratio}.log

            done
        done
    done
done

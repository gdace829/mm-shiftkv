 python build_stats_qwen2vl_multi.py \
   --dataset synthdog \
   --synthdog_json /data/dataset/synthdog-en/synthdog-en.json \
   --model_path /data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac \
   --num_samples 200 --max_decode_tokens 64 \
   --output_dir ./statistics_table/qwen2vl_synthdog


 python build_stats_qwen2vl_multi.py \
   --dataset infographicvqa \
   --parquet_dir /data/model/datasets--lmms-lab--DocVQA/InfographicVQA \
   --split validation \
   --model_path /data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac \
   --num_samples 200 --max_decode_tokens 64 \
   --output_dir ./statistics_table/qwen2vl_infovqa

 python build_stats_qwen2vl_multi.py \
   --dataset docvqa \
   --parquet_dir /data/model/datasets--lmms-lab--DocVQA/DocVQA \
   --split validation \
   --model_path /data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac \
   --num_samples 200 --max_decode_tokens 64 \
   --output_dir ./statistics_table/qwen2vl_docvqa
---
llava
python build_stats_llava_next_multi_dataset.py \
  --pretrained liuhaotian/llava-v1.6-vicuna-7b \
  --dataset docvqa \
  --docvqa_root /data/model/datasets--lmms-lab--DocVQA \
  --docvqa_subset DocVQA \
  --docvqa_split validation \
  --num_samples 200 \
  --max_decode_tokens 64 \
  --output_dir ./statistics_table/llava_next_docvqa

python build_stats_llava_next_multi_dataset.py \
  --pretrained liuhaotian/llava-v1.6-vicuna-7b \
  --dataset docvqa \
  --docvqa_root /data/model/datasets--lmms-lab--DocVQA \
  --docvqa_subset InfographicVQA \
  --docvqa_split validation \
  --num_samples 200 \
  --max_decode_tokens 64 \
  --output_dir ./statistics_table/llava_next_infovqa
--- 

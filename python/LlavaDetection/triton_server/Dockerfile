# RUN apt-get update && apt-get -y install git git-lfs

# Lines to run LLaVA-NeXT

# export MODEL_NAME="llava-v1.6-mistral-7b-hf"
# git clone https://huggingface.co/llava-hf/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

# python ../llama/convert_checkpoint.py \
#     --model_dir tmp/hf_models/${MODEL_NAME} \
#     --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
#     --dtype float16

# trtllm-build \
#     --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
#     --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu \
#     --gpt_attention_plugin float16 \
#     --gemm_plugin float16 \
#     --use_fused_mlp=enable \
#     --max_batch_size 1 \
#     --max_input_len 4096 \
#     --max_seq_len 5120 \
#     --max_num_tokens 4096 \
#     --max_multimodal_len 4096

# python build_visual_engine.py --model_path tmp/hf_models/${MODEL_NAME} --model_type llava_next --model_path tmp/hf_models/${MODEL_NAME} --max_batch_size 5

FROM nvcr.io/nvidia/tritonserver:22.04-py3 as openmpf_triton_server

RUN apt-get update; \
    apt-get -y upgrade; \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade setuptools wheel


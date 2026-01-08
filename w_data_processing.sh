python tools/preprocess_data.py \
    --input /workspace/dataset/redpajama-1t/arxiv/arxiv_fd572627-cce7-4667-a684-fef096dfbeb7.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3-4B \
    --workers 8 \
    --append-eod
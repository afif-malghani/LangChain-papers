from huggingface_hub import hf_hub_download

hf_hub_download(
    "TheBloke/Llama-2-7B-Chat-GGML",
    "llama-2-7b-chat.ggmlv3.q8_0.bin",
    local_dir="."
)
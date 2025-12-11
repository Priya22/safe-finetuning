from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="models/Llama-2-7b-hf"
)

snapshot_download(
    repo_id="PKU-Alignment/beaver-dam-7b",
    local_dir="models/beaver-dam-7b"
)
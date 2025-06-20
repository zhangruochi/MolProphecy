import huggingface_hub
import argparse
import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if __name__ == "__main__":
    # "THUDM/chatglm-6b","meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7b-hf", "bert-base-uncased"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--local_dir",
                        type=str,
                        default="./hugging-hub/pretrained/")
    parser.add_argument("--cache_dir",
                        type=str,
                        default="./hugging-hub/cache/")
    parser.add_argument("--token",
                        type=str,
                        default="")
    parser.add_argument("--force_download",
                        type=bool,
                        default="True")
    
    
    args = parser.parse_args()
    huggingface_hub.login(token=args.token, add_to_git_credential=True)
    huggingface_hub.snapshot_download(args.model,
                                      local_dir_use_symlinks=False,
                                      local_dir=os.path.join(
                                          args.local_dir,
                                          args.model.split("/")[-1]),
                                      cache_dir=args.cache_dir,
                                      token=args.token)

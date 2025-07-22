import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_MAP = {
    'qwen2-vl': {
        '7b': 'Qwen/Qwen2.5-VL-7B-Instruct',  # Latest public Qwen2-VL model
    },
    'llava': {
        '7b': 'llava-hf/llava-1.5-7b-hf',
        # Add more variations if available
    },
    'llama-3': {
        '8b': 'meta-llama/Meta-Llama-3-8B',
        # Add more variations if available
    },
}

def download_model(model_name, variation, save_dir):
    if model_name not in MODEL_MAP or variation not in MODEL_MAP[model_name]:
        print(f"Model or variation not supported: {model_name} {variation}")
        return False
    model_id = MODEL_MAP[model_name][variation]
    model_path = os.path.join(save_dir, model_name + '-' + variation)
    os.makedirs(model_path, exist_ok=True)
    print(f"Downloading {model_id} to {model_path} ...")
    try:
        if model_id.startswith("Qwen/Qwen2.5-VL"):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, cache_dir=model_path, token=HF_TOKEN)
            AutoProcessor.from_pretrained(model_id, cache_dir=model_path, token=HF_TOKEN)
        elif model_id.startswith("llava-hf/llava-1.5-7b-hf"):
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir=model_path, token=HF_TOKEN)
            AutoProcessor.from_pretrained(model_id, cache_dir=model_path, token=HF_TOKEN)
        else:
            AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_path, token=HF_TOKEN)
            AutoTokenizer.from_pretrained(model_id, cache_dir=model_path, token=HF_TOKEN)
        print(f"Download complete: {model_id}")
        return True
    except Exception as e:
        print(f"Failed to download {model_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download open-source models for LongDocURL experiments.")
    parser.add_argument('--model', type=str, help='Comma-separated model names (e.g., qwen2-vl,llava,llama-3)')
    parser.add_argument('--variation', type=str, help='Comma-separated variations (e.g., 7b,7b,8b)')
    parser.add_argument('--save_dir', type=str, default='models/model_files', help='Directory to save the model (default: models/model_files)')
    parser.add_argument('--all', action='store_true', help='Download all models in MODEL_MAP')
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    results = []
    if args.all:
        for model_name in MODEL_MAP:
            for variation in MODEL_MAP[model_name]:
                success = download_model(model_name, variation, save_dir)
                results.append((model_name, variation, success))
    else:
        if not args.model or not args.variation:
            print("You must specify both --model and --variation, or use --all.")
            return
        model_names = [m.strip() for m in args.model.split(",")]
        variations = [v.strip() for v in args.variation.split(",")]
        if len(model_names) != len(variations):
            print("The number of models and variations must match.")
            return
        for model_name, variation in zip(model_names, variations):
            success = download_model(model_name, variation, save_dir)
            results.append((model_name, variation, success))

    print("\nDownload summary:")
    for model_name, variation, success in results:
        status = "Success" if success else "Failed"
        print(f"  {model_name} {variation}: {status}")

if __name__ == '__main__':
    main() 
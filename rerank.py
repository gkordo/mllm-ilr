import json
import pickle
import random

import numpy as np
import os
import yaml
import time
from tqdm import tqdm
import torch
import argparse
from utils import *
from collections import defaultdict

# from transformers import Qwen2_5_VLForConditionalGeneration as QwenReranker
from datasets.generators import get_loaders
from models.rerankers import QwenReranker

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@torch.inference_mode()
def compute_similarity_pair(query_feat, db_img, model, processor, use_confidence):
    db_feat = model.process_image(db_img, processor, model.apply_pq)
    inputs = processor.batch_encode(query_feat, db_feat).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        temperature=None,
    )
    generated_ids = outputs.sequences

    logits = outputs.scores[0]
    selected_logits = logits[
        :, [15, 16]
    ]  # 15 and 16 are the ids for token 0 and token 1
    probs = torch.softmax(selected_logits, dim=-1)
    llm_conf = probs[0][-1].item()
    if use_confidence:
        return llm_conf

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    try:
        return int(response[0].strip())
    except:
        return 0


def compute_similarity(
    query_imgs, query_dict, loader, model, processor, use_confidence, format
):

    similarities = defaultdict(dict)
    pbar = tqdm(loader, desc=f"Reranking {format}...")
    total_batch, start = 0, time.time()
    for batch in pbar:
        start_batch = time.time()
        for db_img, db_id in batch:
            for query in query_dict[db_id]:
                similarities[query][db_id] = compute_similarity_pair(
                    query_imgs[query],
                    db_img,
                    model,
                    processor,
                    use_confidence,
                )
        total_batch += time.time() - start_batch
        sim_per_batch = total_batch / (pbar.n + 1)
        load_per_batch = (time.time() - start) / (pbar.n + 1) - sim_per_batch
        pbar.set_postfix(
            {
                "sim_per_batch": sim_per_batch,
                "load_per_batch": load_per_batch,
            }
        )
    return similarities


@torch.inference_mode()
def main(args):
    config, config_dict = load_config(args.config)

    print("\narguments")
    print("---------------")
    for k, v in sorted(dict(vars(config)).items()):
        print("%s:" % k)
        for kv, vv in sorted(dict(vars(v)).items()):
            print("  %s: %s" % (kv, str(vv)))

    print("\n> loading model")
    model = QwenReranker.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="bfloat16",
        device_map="cuda:0",
        compression_config=config_dict,
        output_hidden_states=False,
        attn_implementation="flash_attention_2",
    )
    processor = Prompter(
        prompt_type=config.preprocessing.prompt_type,
        min_tokens=config.preprocessing.min_tokens,
        max_tokens=config.preprocessing.max_tokens,
    )
    model.eval()

    json_file_path = config.global_similarities.file_path
    with open(json_file_path, "r") as file:
        similarities = json.load(file)

    queries = sorted(similarities.keys())[args.query_start : args.query_end]

    image_dict, shards_dict, query_loader, image_loader, shards_loader = get_loaders(
        queries,
        similarities,
        config.global_similarities.topN,
        config.preprocessing.resolution,
        args,
    )

    use_confidence = config.postprocessing.use_confidence

    query_imgs = {
        id: model.process_image(img, processor, model.apply_pq and not model.apply_asym) for batch in query_loader for (img, id) in batch if id in queries
    }
    print(f"Loaded {len(query_imgs)} queries")

    images_sims = (
        compute_similarity(
            query_imgs,
            image_dict,
            image_loader,
            model,
            processor,
            use_confidence,
            "images",
        )
        if image_dict
        else {}
    )

    shards_sims = (
        compute_similarity(
            query_imgs,
            shards_dict,
            shards_loader,
            model,
            processor,
            use_confidence,
            "shards",
        )
        if shards_dict
        else {}
    )

    output = {q: {**images_sims.get(q, {}), **shards_sims.get(q, {})} for q in queries}

    # New file path to save the modified data
    # Save data to the new file
    new_json_file_path = (
        f"{args.save_dir}/{args.output_name}_{args.query_start}_{args.query_end}.json"
    )
    with open(new_json_file_path, "w") as file:
        json.dump(output, file, indent=4)

    print("File is saved as", new_json_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual similarity with MLLM")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/mnt/data/vrg/ilias/ilias_core/",
        help="Path to the image directory",
    )
    parser.add_argument(
        "--shards_dir",
        type=str,
        default="/mnt/data/vrg/ilias/shards_flickr_recompress/",
        help="Path to the shards directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="rerank/",
        help="Save path for the output json file",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="results",
        help="Output filename of the json file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ilias_siglip_no_comp.yaml",
        help="Path to the run config file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for data loading (default: 128)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading (default: 4)",
    )
    parser.add_argument(
        "--query_start",
        type=int,
        default=0,
        help="Start of the query range (default: 0)",
    )
    parser.add_argument(
        "--query_end",
        type=int,
        default=1232,
        help="End of the query range (default: 1232)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the experiment (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)

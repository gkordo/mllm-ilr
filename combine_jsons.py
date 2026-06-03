import argparse
import os
import json
import glob
import numpy as np
from collections import defaultdict


def main(args):
    ### Json files outputs
    json_files_dir = args.json_files_dir
    output_filename = args.output_filename
    output_json_path = f"{json_files_dir}/{output_filename}.json"

    # === Configuration ===

    # ILIAS
    lambda_ensemble = args.lambda_ensemble

    json_file_path = args.global_similarities
    with open(json_file_path, "r") as file:
        data = json.load(file)

    combined_json = defaultdict(dict)
    # Iterate through each pair of splits
    for json_path in glob.glob(os.path.join(json_files_dir, args.base_filename)):
        # Load current JSON file
        with open(json_path, "r") as f:
            json_data = json.load(f)

        # Add relevant entries
        for q in json_data:
            if q not in combined_json:
                if json_data[q].keys() != data[q].keys():
                    print(f"⚠️ Warning: not all pairs in {json_path} for key {q}")
                for k in json_data[q]:
                    combined_json[q][k] = (
                        lambda_ensemble * data[q][k]
                        + (1 - lambda_ensemble) * json_data[q][k]
                    )
            else:
                print(f"⚠️ Warning: Key {q} found in {json_path}")

    # Save the combined JSON
    with open(output_json_path, "w") as f:
        json.dump(combined_json, f, indent=2)

    print(f"\n✅ Combined JSON saved to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process query range.")
    parser.add_argument(
        "--json_files_dir",
        type=str,
        default="rerank/",
        help="Directory containing JSON files",
    )
    parser.add_argument(
        "--base_filename",
        type=str,
        default="results_*.json",
        help="Base filename for the combined JSON. Supports Unix wildcards.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="results",
        help="Output filename for the combined JSON",
    )
    parser.add_argument(
        "--global_similarities",
        type=str,
        default="similarities_full_i2i_adapt.json",
        help="Global similarities json file",
    )
    parser.add_argument(
        "--lambda_ensemble",
        type=float,
        default=0.5,
        help="Lambda value for ensembling (default: 0.5)",
    )
    args = parser.parse_args()
    main(args)

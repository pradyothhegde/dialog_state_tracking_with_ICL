# python evaluate.py -d -i speech_aware_dialogue/test_dummy_pred_punct.json -g speech_aware_dialogue/test-dstc11.2022-1102.gold.json

# !/usr/bin/env python3
# conding=utf-8

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from metrics import Evaluator

def evaluate(input_data, bleu=False, dst=False, success=False, richness=False, golden="default"):
    """
    Evaluates the given input data using specified metrics.

    Args:
        input_data (dict): The input data to evaluate.
        bleu (bool): If True, BLEU is evaluated.
        dst (bool): If True, dst is evaluated.
        success (bool): If True, inform and success rates are evaluated.
        richness (bool): If True, various lexical richness metrics are evaluated.
        golden (str): Golden file to score against.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    if not bleu and not success and not richness and not dst:
         sys.stderr.write('error: Missing argument, at least one of -b, -d, -s, and -r must be used!\n')
         sys.exit(1)
    e = Evaluator(bleu, success, richness, dst=dst, golden=golden)
    results = e.evaluate(input_data)

    return results

def print_and_save_results(results, output_file):
    """
    Prints the evaluation results to the console and saves them to a JSON file.

    Args:
        results (dict): A dictionary containing the evaluation results.
        output_file (str): Path to the output JSON file.
    """
    for metric, values in results.items():
        if values is not None:
            print(f"====== {metric.upper()} ======")
            for k, v in values.items():
                print(f"{k.ljust(16)}{v}")
            print("")

    with open(output_file, 'w+') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bleu", dest='bleu', action="store_true", default=False, help="If set, BLEU is evaluated.")
    parser.add_argument("-d", "--dst", dest='dst', action="store_true", default=False, help="If set, dst is evaluated.")
    parser.add_argument("-s", "--success", dest='success', action="store_true", default=False, help="If set, inform and success rates are evaluated.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input JSON file path.")
    parser.add_argument("-r", "--richness", dest='richness', action="store_true", default=False, help="If set, various lexical richness metrics are evaluated.")
    parser.add_argument("-o", "--output", type=str, default="evaluation_results.json", help="Output file path, here will be the final report.")
    parser.add_argument("-g", "--golden", type=str, default="default", help="Golden file to score against.")
    
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        input_data = json.load(f)

    results = evaluate(input_data, args.bleu, args.dst, args.success, args.richness, golden='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/new_scripts/mw24/multiwoz_evaluation/speech_aware_dialogue/test-dstc11.2022-1102.gold.json')   # load and evaluate the input data
    # results = evaluate(input_data, False, True, False, False, golden='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/new_scripts/mw24/multiwoz_evaluation/speech_aware_dialogue/test-dstc11.2022-1102.gold.json')   # load and evaluate the input data 
    print_and_save_results(results, args.output)
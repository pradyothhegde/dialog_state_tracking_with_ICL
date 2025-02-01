# export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1); export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1; /homes/kazi/hegde/miniconda3/envs/dialog/bin/python /mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/new_scripts/mw24/data_scripts/new_punctuate.py

from typing import List
from punctuators.models import PunctCapSegModelONNX

def punctuate_and_segment_texts(input_texts: List[str]) -> List[List[str]]:
    """
    Applies punctuation, capitalization, and sentence segmentation to a list of input texts using a pre-trained model.

    Args:
        input_texts: A list of strings to process.

    Returns:
        A list of lists of strings. Each inner list contains the processed sentences for the corresponding input text.
    """
    m: PunctCapSegModelONNX = PunctCapSegModelONNX.from_pretrained(
        "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
    )

    results: List[List[str]] = m.infer(texts=input_texts, apply_sbd=True)
    return results


def remove_punc(s):                                                                                                                               
    puncs = '!"#$%&()*+,./;<=>?@[\\]^_`{|}~' 
    return s.translate(str.maketrans('', '', puncs))

if __name__ == '__main__':
    input_texts = [
        "the us is a nato member as a nato member the country enjoys security guarantees notably article 5",
        "the us is a nhtg member as a nhtg member the country enjoys security guarantees notably article 5",
        "the us is a tuny member as a tuny member the country enjoys security guarantees notably article 5",
        "connor andrew mcdavid is a canadian professional ice hockey centre and captain of the edmonton oilers of the national hockey league the oilers selected him first overall in the 2015 nhl entry draft mcdavid spent his childhood playing ice hockey against older children",
        "please rsvp for the party asap preferably before 8 pm tonight",
    ]


    results = punctuate_and_segment_texts(input_texts)
    
    for input_text, output_texts in zip(input_texts, results):
        print(f"Input: {input_text}")
        print(f"Outputs:")
        for text in output_texts:
            print(f"\t{text}")
        print()
def get_out_file_name(args):
    dataset = args.dataset
    punct = args.punct
    speaker_tag = args.speaker_tag
    slot_placeholder = args.slot_placeholder
    slot_key_sort = args.slot_key_sort
    sentence_embedding_model = args.sentence_embedding_model
    NNcount = args.NNcount
    dialog_history = args.dialog_history
    decoding = args.decoding

    if punct == "O":
        punct_ret = "OP"
    elif punct == "N":
        punct_ret = "NP"
    elif punct == "M":
        punct_ret = "MP"

    if speaker_tag == "Y":
        speaker_tag_ret = "ST"
    else:
        speaker_tag_ret = "NST"

    if slot_placeholder == "not mentioned":
        slot_placeholder_ret = "PH-nm"
    elif slot_placeholder == "N.A.":
        slot_placeholder_ret = "PH-NA"
    elif slot_placeholder == "none":
        slot_placeholder_ret = "PH-none"
    elif slot_placeholder == "omit":
        slot_placeholder_ret = "PH-omit"
    elif slot_placeholder == "empty":
        slot_placeholder_ret = "PH-empty"

    if slot_key_sort == "Y":
        slot_key_sort_ret = "SO"
    elif slot_key_sort == "N":
        slot_key_sort_ret = "SU"
    elif slot_key_sort == "1":
        slot_key_sort_ret = "SU1"
    elif slot_key_sort == "2":
        slot_key_sort_ret = "SU2"

    if sentence_embedding_model == "sentence-transformers/LaBSE":
        sentence_embedding_model_ret = "Labse"
    else: 
        sentence_embedding_model_ret = "D2F"

    file_name = f"{dataset}_{punct_ret}_{speaker_tag_ret}_{slot_placeholder_ret}_{slot_key_sort_ret}_{sentence_embedding_model_ret}_NN-{NNcount}_{dialog_history}_{decoding}"

    return file_name

# Write a detailed description of the function here:
# Dataset = "MW24"
# Punct = original - "O" | no punctuation - "N" | model punctuation - "M"
# Speaker_tag = "Y" | "N"
# Slot_placeholder = "not mentioned" | "N.A." | "none" | deleting the slot key if there is no placeholder - "omit" | empty string - "empty"
# Slot_key_sort = "Y" | "N" | seed - "1" | "2"
# Sentence_embedding_model = "sentence-transformers/LaBSE" 
# NNcount = 10
# Dialog_history = "Y" | "N"
# Decoding = slot key and value given domain - "SKV" | slot value given slot key - "SV"
def append_model_name(output_folder_name, model_name):
    if model_name == 'allenai/OLMo-7B-Instruct':
        output_folder_name = 'O7BI_' + output_folder_name
    elif model_name == 'mistralai/Mistral-7B-Instruct-v0.3':
        output_folder_name = 'M7BI03_' + output_folder_name
    else:
        print('Model name not recognized, go to append_model_name.py to add the model name')
        exit()
    return output_folder_name
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


import whisper
import argparse

from joblib import Parallel, delayed
# import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import json

from jiwer import wer
import re
import pandas as pd

def replace_numbers(text):
    number_mapping = {
        '0': ' zero', '1': ' one', '2': ' two', '3': ' three', '4': ' four',
        '5': ' five', '6': ' six', '7': ' seven', '8': ' eight', '9': ' nine'
    }
    for digit, word in number_mapping.items():
        text = text.replace(digit, word)

    return text


def read(pred_path, filename, model):
    spk_name, filename = filename.split("/")[-2:]
    GT_path = os.path.join("/cm/archive/sonnn45/Dubbing_Dataset/GRID/Data_and_Feature_GRID/Grid_wav_RawTxt", spk_name, filename.replace(".wav", ".lab"))
    result = model.transcribe(pred_path)
    predicted = result["text"]

    predicted_line = predicted.strip()
    
    if predicted_line:
        text_lines = []
        with open(GT_path, 'r', encoding='utf-8') as file:  
            for line in file:
                cleaned_line = line.rstrip()
                text_lines.append(cleaned_line)
        text_content = ' '.join(text_lines)

        text_content = re.sub(r'[^\w\s]', '', text_content)
        text_content = replace_numbers(text_content)
        predicted = re.sub(r'[^\w\s]', '', predicted)
        predicted = replace_numbers(predicted)

        error = wer(predicted.lower(), text_content.lower())
        return error, filename, predicted.lower(), text_content.lower()
            
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pred_path",
        type=str,
        required=True,
        help="path to pred wav files",
    )

    parser.add_argument(
        "-n",
        "--dataset_name",
        type=str,
        required=True,
        help="dataset name"
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="cuda or cpu"

    )

    parser.add_argument(
        "-m", 
        "--metadata_path", 
        type=str,
        required=True, 
        help="path to the metadata file",
    )

    args = parser.parse_args()

    if args.dataset_name == "GRID":
        model_name = "base"
    elif args.dataset_name == "V2C":
        model_name = "large-v3"
    else:
        NotImplementedError(f"{args.dataset_name} dataset is not implemented")

    model = whisper.load_model(model_name, device=args.device)
    df = pd.read_csv(args.metadata_path, sep="\t")

    with open(args.metadata_path, "r") as f:
        metadata = json.load(f)

    wav_list = [item["audio_path"] for item in metadata]

    results_P = []
    for i in tqdm(wav_list):
        filename = os.path.basename(i)
        pred_path = os.path.join(args.pred_path, filename)

        try:
            results_P.append(read(pred_path, i, model))
        except Exception:
            print("{} is not normal".format(i))
            continue

    results = [i[0] for i in results_P if i is not None and i[0] is not None]
    nonename = [i[1] for i in results_P if i is not None and i[0] is not None]
    predicted_text = [i[2] for i in results_P if i is not None and i[0] is not None]
    gt_text = [i[3] for i in results_P if i is not None and i[0] is not None]
    

    print("len(results)", len(results))
    print("len(nonename)", len(nonename))

    print("The test path: {}".format(args.pred_path))
    print("SUM:", sum(results), "Length:", len(results), "WER_Currtly:", sum(results)/len(results), "None_number: ", len(wav_list)-len(results))
    print("============================over=========================")
    print("\n")

    exp_name = os.path.basename(args.pred_path)
    output_file_path = os.path.join('wer_log', '{}.txt'.format(exp_name))

    with open(output_file_path, 'a') as file:
        for i in range(len(results)):
            file.write("{}: WER-Result:{:.4f}, gt_content:{}, predicted_content:{} \n".format(
                    nonename[i], results[i], gt_text[i], predicted_text[i]
            ))
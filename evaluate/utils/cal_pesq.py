from pesq import pesq
import pandas as pd
import os
import librosa
import glob
from tqdm import tqdm
import fire

def cal_pesq(
    synth_path,
    prompt_path,
    output_file,
    target2prompt,
    sr=16000,
    ):

    results = []
    synth_files = sorted(glob.glob(os.path.join(synth_path, "*.wav"), recursive=True))
    for syns_file in tqdm(synth_files):        
        try:
            filename = os.path.basename(syns_file)
            prompt_name = target2prompt[filename]
            prompt_file = os.path.join(prompt_path, prompt_name)

            prompt_wav, _ = librosa.load(prompt_file, sr=sr)
            synth_wav, _ = librosa.load(syns_file, sr=sr)

            pesq_nb = pesq(sr, prompt_wav, synth_wav, "nb")
            pesq_wb = pesq(sr, prompt_wav, synth_wav, "wb")
            results.append([filename, pesq_nb, pesq_wb])
        except:
            continue

    df = pd.DataFrame(results, columns=['Filename', 'PESQ_NB', "PESQ_WB"])
    df.to_csv(output_file)

    pesq_nb_mean = df["PESQ_NB"].mean()
    pesq_wb_mean = df["PESQ_WB"].mean()
    print('='*100, '\nPESQ WB :', pesq_wb_mean, '\nPESQ NB :', pesq_nb_mean)

    return {
        "WB": pesq_wb_mean,
        "NB": pesq_nb_mean,
    }


if __name__ == "__main__":
    fire.Fire(cal_pesq)
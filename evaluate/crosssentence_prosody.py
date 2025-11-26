import os
import glob
import fire
import librosa
import parselmouth
import numpy as np
import pandas as pd
import pyworld as pw
from tqdm import tqdm
import argparse


def gets_mean_pitch(wav_path):
    x, fs = librosa.load(wav_path)
    x = x.astype(np.double)
    _f0_h, t_h = pw.dio(x, fs)

    f0_h = pw.stonemask(x, _f0_h, t_h, fs)
    data = f0_h
    mask = (data != 0)

    segments = np.split(data, np.where(np.diff(mask))[0]+1)
    non_zero_num,value = 0, 0
    for seg in segments:
        if len(seg)<=0 or seg[0]==0:
            continue
        non_zero_num += len(seg)
        value += seg.sum()

    if non_zero_num==0:
      return 0, False
    
    return value/non_zero_num, True


def get_energy(wav_path):
    y, sr = librosa.load(wav_path)
    energy = librosa.feature.rms(y=y)
    energy = energy.mean()
    return energy


def get_pitch_energy(wav_path):
    pitch, _ = gets_mean_pitch(wav_path)
    energy = get_energy(wav_path)
    return pitch, energy


###  pitch [136.57698522, 196.09780757]   < low, normal, > high
def pitch_rank(pitch):
    if pitch < 136.57698522:
        return 0
    elif pitch > 196.09780757:
        return 2
    else:
        return 1


### energy [0.03331899, 0.05054203]  < 0.03331899 low, normal, > 0.05054203 high
def energy_rank(energy):
    if energy < 0.03331899:
        return 0
    elif energy > 0.05054203:
        return 2
    else:
        return 1


def cal_prosody_list(f0_mean, f0_rank, e_mean, e_rank):
    f0_rmse_list, f0_compare_list, e_rmse_list, e_compare_list = [], [], [], []
    for i in range(len(f0_mean) - 1):
        for j in range(i + 1, len(f0_mean)):
            f0_rmse = np.sqrt((f0_mean[i] - f0_mean[j])**2)
            e_rmse = np.sqrt((e_mean[i] - e_mean[j])**2)
            f0_compare = 1 if f0_rank[i] == f0_rank[j] else 0
            e_compare = 1 if e_rank[i] == e_rank[j] else 0

            f0_rmse_list.append(f0_rmse)
            f0_compare_list.append(f0_compare)
            e_rmse_list.append(e_rmse)
            e_compare_list.append(e_compare)
    return np.mean(f0_rmse_list), np.mean(f0_compare_list), np.mean(e_rmse_list), np.mean(e_compare_list)


def cal_prosody(
    synth_path,
    output_path,
    sr=16000,
    ):

    synth_files = sorted(glob.glob(os.path.join(synth_path, "*.wav"), recursive=True))

    # Group by prompt
    group_by_prompt = {}
    for fp in synth_files:
        filename = os.path.basename(fp)
        prompt_filename = filename.split('_')[0]
        if prompt_filename not in group_by_prompt:
            group_by_prompt[prompt_filename] = []
        group_by_prompt[prompt_filename].append(fp)

    results = {}
    for prompt_filename in tqdm(list(group_by_prompt.keys())):
        results[prompt_filename] = {}
        results[prompt_filename]['F0-Mean'] = []
        results[prompt_filename]['F0-Rank'] = []
        results[prompt_filename]['E-Mean'] = []
        results[prompt_filename]['E-Rank'] = []

        for fp in group_by_prompt[prompt_filename]:
            f0_synth_mean, energy_synth_mean = get_pitch_energy(fp)
            f0_synth_rank = pitch_rank(f0_synth_mean)
            energy_synth_rank = energy_rank(energy_synth_mean)

            results[prompt_filename]['F0-Mean'].append(f0_synth_mean)
            results[prompt_filename]['F0-Rank'].append(f0_synth_rank)
            results[prompt_filename]['E-Mean'].append(energy_synth_mean)
            results[prompt_filename]['E-Rank'].append(energy_synth_rank)

    final_results = []
    for prompt_filename in tqdm(list(results.keys())):
        f0_rmse_mean, f0_compare_mean, e_rmse_mean, e_compare_mean = cal_prosody_list(
            results[prompt_filename]['F0-Mean'],
            results[prompt_filename]['F0-Rank'],
            results[prompt_filename]['E-Mean'],
            results[prompt_filename]['E-Rank']
        )
        final_results.append([
            prompt_filename, 
            f0_rmse_mean, 
            f0_compare_mean, 
            e_rmse_mean, 
            e_compare_mean
        ])

    df = pd.DataFrame(final_results, columns=[
        'Prompt', 
        'F0-RMSE-MEAN', 
        'F0-COMPARE-MEAN', 
        'ENERGY-RMSE-MEAN',         
        'ENERGY-COMPARE-MEAN', 
    ])
    df.to_csv(output_path, index=False)

    f0_rmse_mean = df["F0-RMSE-MEAN"].mean()
    f0_acc = df['F0-COMPARE-MEAN'].mean()
    energy_rmse_mean = df["ENERGY-RMSE-MEAN"].mean()
    energy_acc = df['ENERGY-COMPARE-MEAN'].mean()
    print('='*100, '\nF0:', f0_acc, f0_rmse_mean, '\nENERGY :', energy_acc, energy_rmse_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth_path', required=True)
    parser.add_argument('--output_path', required=True)

    args = parser.parse_args()
    synth_path = args.synth_path
    output_path = args.output_path

    cal_prosody(
        synth_path=synth_path,
        output_path=output_path,
    )
import os
import glob
import fire
import librosa
import parselmouth
import numpy as np
import pandas as pd
import pyworld as pw
from tqdm import tqdm


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


def cal_prosody(
    synth_path,
    prompt_path,
    output_file,
    target2prompt,
    sr=16000,
    ):

    results = []
    synth_files = sorted(glob.glob(os.path.join(synth_path, "*.wav"), recursive=True))

    for synth_file in tqdm(synth_files):
        try:
            filename = os.path.basename(synth_file)
            prompt_name = target2prompt[filename]
            prompt_file = os.path.join(prompt_path, prompt_name)

            f0_prompt_mean, energy_prompt_mean = get_pitch_energy(prompt_file)
            f0_synth_mean, energy_synth_mean = get_pitch_energy(synth_file)

            f0_prompt_rank = pitch_rank(f0_prompt_mean)
            f0_synth_rank = pitch_rank(f0_synth_mean)
            energy_prompt_rank = energy_rank(energy_prompt_mean)
            energy_synth_rank = energy_rank(energy_synth_mean)

            results.append([
                filename, 
                f0_prompt_mean, 
                f0_synth_mean, 
                energy_prompt_mean, 
                energy_synth_mean,
                f0_prompt_rank,
                f0_synth_rank,
                energy_prompt_rank,
                energy_synth_rank,
            ])
        except:
            continue

    df = pd.DataFrame(results, columns=[
        'Filename', 
        'F0-PROMPT', 
        'F0-SYNTH', 
        'ENERGY-PROMPT', 
        'ENERGY-SYNTH',
        'F0-PROMPT-RANK', 
        'F0-SYNTH-RANK', 
        'ENERGY-PROMPT-RANK', 
        'ENERGY-SYNTH-RANK',
    ])

    df['F0-RMSE'] = np.sqrt((df['F0-PROMPT'] - df['F0-SYNTH'])**2)
    df['ENERGY-RMSE'] = np.sqrt((df['ENERGY-PROMPT'] - df['ENERGY-SYNTH'])**2)
    df['F0-COMPARE'] = (df['F0-PROMPT-RANK'] == df['F0-SYNTH-RANK']).astype(int)
    df['ENERGY-COMPARE'] = (df['ENERGY-PROMPT-RANK'] == df['ENERGY-SYNTH-RANK']).astype(int)
    
    df.to_csv(output_file, index=False)

    f0_rmse_mean = df["F0-RMSE"].mean()
    f0_acc = df['F0-COMPARE'].mean()
    energy_rmse_mean = df["ENERGY-RMSE"].mean()
    energy_acc = df['ENERGY-COMPARE'].mean()
    print('='*100, '\nF0:', f0_acc, f0_rmse_mean, '\nENERGY :', energy_acc, energy_rmse_mean)

    return {
        "F0 ACC": f0_acc,
        "F0 RMSE": f0_rmse_mean,
        "ENERGY ACC": energy_acc,
        "ENERGY RMSE": energy_rmse_mean,
    }

if __name__ == "__main__":
    fire.Fire(cal_prosody)
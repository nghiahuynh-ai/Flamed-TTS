import os
import tgt
import fire
import glob
import librosa
import pandas as pd
from tqdm import tqdm
from typing import Optional


def cal_phone(
    tgt_path,
    output_file,
    target2transcript,
    sr=16000,
    ):
    tgt_files = sorted(glob.glob(os.path.join(tgt_path, "*.TextGrid")))

    results = []
    for tgt_file in tqdm(tgt_files):
        phone_dur, sil_dur, pauses = [], [], 0
        try:
            filename = os.path.basename(tgt_file)
            textgrid = tgt.io.read_textgrid(tgt_file, include_empty_intervals=True)
            textgrid_tier = textgrid.get_tier_by_name("phones")
            for t in textgrid_tier._objects:
                s, e, p = t.start_time, t.end_time, t.text
                if p != "":
                    phone_dur.append(e-s)
                else:
                    pauses += 1
                    sil_dur.append(e-s)
        except:
            continue

        try:
            phone_dur_mean = sum(phone_dur) / len(phone_dur)
        except:
            phone_dur_mean = 0
        
        try:
            sil_dur_mean = sum(sil_dur) / len(sil_dur)
        except:
            sil_dur_mean = 0

        results.append([filename, phone_dur_mean, pauses, sil_dur_mean])

    df = pd.DataFrame(results, columns=['File', 'MPhD', 'Pauses', 'MPaD'])
    df.to_csv(output_file)

    mphd_mean, mphd_std = df['MPhD'].mean(), df['MPhD'].std() 
    pauses_mean, pauses_std = df['Pauses'].mean(), df['Pauses'].std() 
    mpad_mean, mpad_std = df['MPaD'].mean(), df['MPaD'].std() 
    print('='*100, '\nPhoneme :', mphd_mean, '±', mphd_std, pauses_mean, '±', pauses_std, mpad_mean, '±', mpad_std)

    return [mphd_mean,  mphd_std, pauses_mean, pauses_std, mpad_mean, mpad_std]

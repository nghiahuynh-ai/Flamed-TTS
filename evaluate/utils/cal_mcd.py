from pymcd.mcd import Calculate_MCD
import pandas as pd
import os
import glob
from tqdm import tqdm
import fire


def cal_mcd(
    synth_path,
    prompt_path,
    output_file,
    target2prompt,
    mcd_mode = 'dtw'    
    ):
    """
    MCD (plain): the conventional MCD metric, which requires the lengths of two input speeches to be the same. Otherwise, it would simply extend the shorted speech to the length of longer one by padding zero for the time-domain waveform.
    MCD-DTW: an improved MCD metric that adopts the Dynamic Time Warping (DTW) algorithm to find the minimum MCD between two speeches.
    MCD-DTW-SL: MCD-DTW weighted by Speech Length (SL) evaluates both the length and the quality of alignment between two speeches. Based on the MCD-DTW metric, the MCD-DTW-SL incorporates an additional coefficient w.r.t. the difference between the lengths of two speeches.
    """
    assert mcd_mode in ['plain', 'dtw', 'dtw_sl']
    mcd_calculator = Calculate_MCD(mcd_mode)

    results = []
    synth_files = sorted(glob.glob(os.path.join(synth_path, "*.wav"), recursive=True))
    for synth_file in tqdm(synth_files):
        filename = os.path.basename(synth_file)
        promptname = target2prompt[filename]
        prompt_file = os.path.join(prompt_path, promptname)
        mcd = mcd_calculator.calculate_mcd(prompt_file, synth_file)
        results.append([prompt_file, mcd])

    df = pd.DataFrame(results, columns=['File', 'MCD'])
    df.to_csv(output_file)
    mcd_mean = df['MCD'].mean()
    print('='*100, '\nMCD :', mcd_mean)
    return mcd_mean


if __name__ == "__main__":
    fire.Fire(cal_mcd)

    
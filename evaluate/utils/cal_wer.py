import fire
import glob
import os
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from torchmetrics.functional.text import word_error_rate
from transformers import Wav2Vec2Processor, HubertForCTC


def cal_wer(
    audio_path,
    output_file,
    target2transcript,
    sr=16000,
    device='cuda:0',
    model_ckpt="facebook/hubert-large-ls960-ft",
    processor="facebook/hubert-large-ls960-ft",
    ):

    processor = Wav2Vec2Processor.from_pretrained(processor)
    model = HubertForCTC.from_pretrained(model_ckpt).to(device)
    
    audios = sorted(glob.glob(os.path.join(audio_path, "*.wav")))

    results = []
    for audio_file in tqdm(audios):
        try:
            filename = os.path.basename(audio_file)
            text = target2transcript[filename]
            audio, _ = librosa.load(audio_file, sr=sr)

            input_values = processor(audio=audio, sampling_rate=sr, return_tensors="pt").input_values.to(device)  # Batch size 1
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0]).lower().strip()

            wer = word_error_rate(transcription, text)
            results.append([filename, text, transcription, wer.item()])
        except:
            continue

    df = pd.DataFrame(results, columns=['File', 'Prompt', 'Synthesized', 'WER'])
    df.to_csv(output_file)

    wer_mean = df['WER'].mean()
    print('='*100, '\nWER :', wer_mean)

    return wer_mean
        

if __name__ == "__main__":
    fire.Fire(cal_wer)
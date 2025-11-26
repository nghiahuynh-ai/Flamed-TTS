import glob
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import librosa
import torch
import fire
import torch.nn.functional as F
from models.speaker.ecapa_tdnn import ECAPA_TDNN_SMALL
import argparse
import torchaudio
import torchaudio.transforms as T


device = 'cuda' if torch.cuda.is_available() else "cpu"
MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]


def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


def cal_sim_list_of_emb(list_input):
    cosine_sim = []
    for i in range(len(list_input) - 1):
        for j in range(i + 1, len(list_input)):
            sim = F.cosine_similarity(list_input[i], list_input[j])
            sim = sim[0].item()
            cosine_sim.append(sim)
    return np.mean(cosine_sim)


def cal_sim(
        synth_path,
        output_path,
        sr=16000,
        device='cuda:0',
        model_name='wavlm_large',
        ckpt="./ckpt/wavlm_large_nofinetune.pth"
    ):
    
    model = init_model(model_name, ckpt)
    model = model.to(device)
    model.eval()

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
        results[prompt_filename] = []

        for fp in group_by_prompt[prompt_filename]:
            filename = os.path.basename(fp)

            # synth_wav, _ = librosa.load(path=fp, sr=sr)
            synth_wav, orig_sr = torchaudio.load(fp)
            if orig_sr != sr:
                resampler = T.Resample(orig_freq=orig_sr, new_freq=sr)
                synth_wav = resampler(synth_wav)

            # synth_wav = torch.from_numpy(synth_wav).unsqueeze(0).float()
            synth_wav = synth_wav.to(device)
            with torch.no_grad():
                synth_emb = model(synth_wav)
            # synth_emb = synth_emb.detach().cpu().numpy()
            # results.append([filename, prompt_filename, synth_emb])
            results[prompt_filename].append(synth_emb)
    
    final_results = []
    for prompt_filename in tqdm(list(results.keys())):
        cosine_sim = cal_sim_list_of_emb(results[prompt_filename])
        final_results.append([prompt_filename, cosine_sim])

    df = pd.DataFrame(final_results, columns=['Prompt', 'Sim'])
    df.to_csv(output_path, index=False)
    print('='*100, '\n', 'Cross-sentence SIM :', df['Sim'].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()
    synth_path = args.synth_path
    output_path = args.output_path
    device = args.device

    cal_sim(
        synth_path=synth_path,
        output_path=output_path,
        device=device,
    )

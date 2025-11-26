import glob
import os
from pathlib import Path

import fire
import librosa
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.facodec.infer import fa_decoder, fa_encoder
from models.speaker.ecapa_tdnn import ECAPA_TDNN_SMALL
from utils.checkpoints import resolve_checkpoint
DEFAULT_SIM_CKPT = os.environ.get("SIM_CKPT", "ckpt/wavlm_large_nofinetune.pth")
DEFAULT_SIM_MODEL = os.environ.get("SIM_MODEL", "wavlm_large")
DEFAULT_CACHE_DIR = Path(os.environ.get("SIM_CACHE", Path.home() / ".cache" / "flamed" / "sim")).expanduser()

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


def facodec(wav, device='cuda:0'):
    global fa_encoder, fa_decoder
    fa_encoder = fa_encoder.to(device)
    fa_decoder = fa_decoder.to(device)

    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # encode
        enc_out = fa_encoder(wav)
        # quantize
        vq_post_emb, _, _, _, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
        # decode (recommand)
        recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
        recon_wav = recon_wav[0][0].cpu().numpy()

    return recon_wav


def encodec(wav, sr, device='cuda:0'):
    global encodec_model, encodec_processor
    encodec_model.to(device)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=encodec_processor.sampling_rate)
    inputs = encodec_processor(raw_audio=wav, sampling_rate=encodec_processor.sampling_rate, return_tensors="pt")
    encoder_outputs = encodec_model.encode(inputs["input_values"].to(device), inputs["padding_mask"].to(device))
    recon_wav = encodec_model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
    recon_wav = recon_wav.squeeze().detach().cpu().numpy()
    recon_wav = librosa.resample(recon_wav, orig_sr=encodec_processor.sampling_rate, target_sr=sr)
    return recon_wav


def speechtokenizer(wav, sr, device='cuda:0'):
    global speechtokenizer_model
    speechtokenizer_model.to(device)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=speechtokenizer_model.sample_rate)
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(1).to(device)
    with torch.no_grad():
        codes = speechtokenizer_model.encode(wav) # codes: (n_q, B, T)
    recon_wav = speechtokenizer_model.decode(codes)
    recon_wav = recon_wav.squeeze().detach().cpu().numpy()
    recon_wav = librosa.resample(recon_wav, orig_sr=speechtokenizer_model.sample_rate, target_sr=sr)
    return recon_wav


def cal_sim(
        synth_path,
        prompt_path,
        output_file,
        target2prompt,
        codec,
        sr=16000,
        device='cuda:0',
        model_name=DEFAULT_SIM_MODEL,
        ckpt=DEFAULT_SIM_CKPT,
        cache_dir: str | Path | None = None,
    ):
    
    ckpt_path = resolve_checkpoint(ckpt, cache_dir or DEFAULT_CACHE_DIR) if ckpt else None
    model = init_model(model_name, str(ckpt_path) if ckpt_path else None).to(device)
    model.eval()

    results = []
    synth_files = sorted(glob.glob(os.path.join(synth_path, "*.wav"), recursive=True))

    for synth_file in tqdm(synth_files):
        try:
            filename = os.path.basename(synth_file)
            prompt_name = target2prompt[filename]
            prompt_file = os.path.join(prompt_path, prompt_name)

            synth_wav, _ = librosa.load(path=synth_file, sr=sr)
            prompt_wav, _ = librosa.load(path=prompt_file, sr=sr)

            codec_lower = codec.lower()
            if codec_lower == 'facodec':
                recon_wav = facodec(prompt_wav, device)
            elif codec_lower == 'encodec':
                recon_wav = encodec(prompt_wav, sr, device)
            elif codec_lower == 'speechtokenizer':
                recon_wav = speechtokenizer(prompt_wav, sr, device)
            else:
                raise ValueError(f"Unsupported codec: {codec}")

            synth_wav = torch.from_numpy(synth_wav).unsqueeze(0).float().to(device)
            prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0).float().to(device)
            recon_wav = torch.from_numpy(recon_wav).unsqueeze(0).float().to(device)

            with torch.no_grad():
                synth_emb = model(synth_wav)
                promt_emb = model(prompt_wav)
                recon_emb = model(recon_wav)

            sim = F.cosine_similarity(synth_emb, promt_emb)
            sim = sim[0].item()

            sim_r = F.cosine_similarity(synth_emb, recon_emb)
            sim_r = sim_r[0].item()

            results.append([filename, sim, sim_r])
        except Exception as exc:
            print(f"[SIM] Skipped {synth_file}: {exc}")
            continue

    # Convert the 2D array to a Pandas DataFrame
    df = pd.DataFrame(results, columns=['File', 'SIM-O', "SIM-R"])

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

    sim_o_mean = df["SIM-O"].mean()
    sim_r_mean = df["SIM-R"].mean()
    print('='*100, '\nSIM-O :', sim_o_mean, '\nSIM-R :', sim_r_mean)

    return {
        "SIM-O": sim_o_mean,
        "SIM-R": sim_r_mean
    }

if __name__ == "__main__":
    fire.Fire(cal_sim)

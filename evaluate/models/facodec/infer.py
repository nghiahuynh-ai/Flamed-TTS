from models.facodec.ns3_codec.facodec import FACodecEncoder, FACodecDecoder
from transformers import EncodecModel, AutoProcessor
from huggingface_hub import hf_hub_download
import torch
# from speechtokenizer import SpeechTokenizer

fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

# fa_encoder.load_state_dict(torch.load("ckpt/ns3_facodec_encoder.bin"))
# fa_decoder.load_state_dict(torch.load("ckpt/ns3_facodec_decoder.bin"))

encoder_ckpt = hf_hub_download(
        repo_id="amphion/naturalspeech3_facodec", 
        filename="ns3_facodec_encoder.bin"
    )
decoder_ckpt = hf_hub_download(
    repo_id="amphion/naturalspeech3_facodec", 
    filename="ns3_facodec_decoder.bin"
)
fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

fa_encoder.eval()
fa_decoder.eval()

encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
encodec_model.eval()

# config_path = "/cm/archive/sonnn45/Amphion/ckpts/tts/speechtokenizer_hubert_avg/config.json"
# ckpt_path = "/cm/archive/sonnn45/Amphion/ckpts/tts/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
# speechtokenizer_model = SpeechTokenizer.load_from_checkpoint(
#     config_path, ckpt_path
# )
# speechtokenizer_model.eval()
import os
import re
import time
import torch
import librosa
import numpy as np
from g2p_en import G2p
from string import punctuation
from omegaconf import DictConfig
from flamed.text import text_to_sequence
from flamed.models.flamed_lightning import FlamedLightning
from flamed.models.synthesizer import (
    PriorGenerator,
    ProbGenerator,
)
from flamed.models.facodec import (
    FACodecEncoder,
    FACodecDecoder,
)


class Flamed(FlamedLightning):
    
    @classmethod
    def from_pretrained(
        cls,
        cfg,
        ckpt_path,
        device,
        weights_only=False,
        training_mode=False,
        modules=None,
    ):
        model = Flamed(cfg)
        model.lexicon = model.read_lexicon()
        model.g2p = G2p()

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=weights_only)
        state = ckpt['state_dict'] if (not weights_only and 'state_dict' in ckpt) else ckpt

        prefixes = None
        if modules:
            prefixes = tuple(f"{m}." for m in modules)
            state = {k: v for k, v in state.items() if k.startswith(prefixes)}
            if not state:
                raise ValueError(f"No weights found for modules {modules} in checkpoint: {ckpt_path}")

        load_result = model.load_state_dict(state, strict=modules is None)
        if prefixes:
            missing_relevant = [k for k in load_result.missing_keys if k.startswith(prefixes)]
            if missing_relevant:
                raise ValueError(
                    f"Missing expected weights for modules {modules} in checkpoint {ckpt_path}: {missing_relevant}"
                )
        del ckpt

        if not training_mode:
            model.eval()
        return model
    
    def __init__(self, cfg):
        super(Flamed, self).__init__()

        self.cfg = cfg
        self.pipeline = tuple(self._prepare_pipeline(cfg.get('pipeline')))
        self._train_prior = 'PriorGenerator' in self.pipeline
        self._train_prob = 'ProbGenerator' in self.pipeline
        self.prior_generator = PriorGenerator(cfg['prior_generator'])
        self.prob_generator = ProbGenerator(cfg['prob_generator']) if self._train_prob else None
        self._apply_pipeline_freezing()
    
    def _prepare_pipeline(self, pipeline_cfg):
        if pipeline_cfg is None:
            return ['PriorGenerator', 'ProbGenerator']
        if isinstance(pipeline_cfg, str):
            pipeline_cfg = [pipeline_cfg]
        seen = set()
        ordered = []
        for module in pipeline_cfg:
            name = str(module)
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered
    
    def _apply_pipeline_freezing(self):
        if not self._train_prior:
            self._freeze_module(self.prior_generator)
        if self.prob_generator is not None and not self._train_prob:
            self._freeze_module(self.prob_generator)
    
    @staticmethod
    def _freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen modules in eval mode even during model.train()
        if not self._train_prior:
            self.prior_generator.eval()
        if self.prob_generator is not None and not self._train_prob:
            self.prob_generator.eval()
        return self
        
    def forward(
        self,
        phonemes, 
        x_len, 
        codes, 
        y_len, 
        phone_durations,
        sil_durations,
        embs,
        prompts,
        spks,
        training=True,
        ):
        
        # Run PriorGenerator; detach gradients when it is frozen.
        prior_training = training and self._train_prior
        prior_mode_flag = training and self._train_prior
        prior_context = torch.enable_grad() if prior_training else torch.no_grad()
        with prior_context:
            (
                prior_embs,
                tgt_masks,
                ar_losses,
            ) = self.prior_generator.compute_loss(
                texts=phonemes,
                src_lens=x_len,
                max_src_len=phonemes.size(-1),
                codes=codes,
                tgt_lens=y_len,
                max_tgt_len=codes.size(-1),
                phone_durations=phone_durations,
                sil_durations=sil_durations,
                prompts=prompts,
                prompts_len=prompts.size(-1),
                training=prior_mode_flag,
            )
        losses = {}
        if self._train_prior:
            losses.update(ar_losses)

        cond_embs = prior_embs if prior_training else prior_embs.detach()

        # Forward & compute losses of flow matching when enabled.
        if self._train_prob and self.prob_generator is not None:
            prob_losses = self.prob_generator.compute_loss(
                x1=embs,
                cond=cond_embs,
                spk=spks,
                mask=~tgt_masks.unsqueeze(-1),
                training=training,
            )
            losses.update(prob_losses)
        
        return losses
    
    @torch.inference_mode()
    def sample(
        self, 
        text: str = None,
        phonemes: torch.Tensor = None,
        prompt_raw: str | np.ndarray | torch.Tensor = None,
        prompt_processed: torch.Tensor = None,
        timbre: torch.Tensor = None,
        sr: int = 16000,
        codec_cfg: DictConfig = None,
        codec_encoder: torch.nn.Module = None,
        codec_decoder: torch.nn.Module = None,
        temp_durgen: float = 0.3,
        temp_denoiser: float = 0.3,
        nsteps_durgen: int = 64,
        nsteps_denoiser: int = 64,
        guidance_scale: float | None = None,
        lexicon_path: str = None,
        cleaners: str = ['english_cleaners'],
        ):
        
        if self.prob_generator is None:
            raise ValueError('ProbGenerator is not initialized (pipeline excluded ProbGenerator); sampling is unavailable.')
        
        if codec_encoder is None or codec_decoder is None:
            if codec_cfg is None:
                raise ValueError('The codec_encoder or codec_decoder is set to None. To initialize the codec encoder or decoder, you need to provide a codec_cfg of type omegaconf.DictConfig.')
            codec_encoder, codec_decoder = self._get_codec_models(codec_cfg)

        text_provided = text is not None and phonemes is None
        phonemes_provided = text is None and phonemes is not None
        text_phoneme_exclusive_check =  text_provided or phonemes_provided
        if not text_phoneme_exclusive_check:
            raise ValueError('`text` and `phonemes` are mutually exclusive—only one should be provided, and the other must be None!')

        prompt_raw_provided = prompt_raw is not None and prompt_processed is None
        prompt_processed_provided = prompt_raw is None and prompt_processed is not None
        prompt_raw_processed_exclusive_check = prompt_raw_provided or prompt_processed_provided
        if not prompt_raw_processed_exclusive_check:
            raise ValueError('`prompt_raw` and `prompt_processed` are mutually exclusive—only one should be provided, and the other must be None!')

        # get starting timestamp of the progress
        start_time = time.time()
        
        # process phoneme
        if text_provided:
            phonemes, _, _ = self._preprocess_english(text, lexicon_path, cleaners)
        else:
            phonemes = phonemes.unsqueeze(0).to(self.device)
        
        # process acoustic prompt
        if prompt_raw_provided:
            acoustic_prompt = self._preprocess_acoustic_prompt(prompt_raw, sr)
            enc_out = codec_encoder(acoustic_prompt)
            _, prompts, _, _, timbre = codec_decoder(enc_out, eval_vq=False, vq=True)
            prompts = prompts.permute(1, 0, 2)
        else:
            if timbre is None:
                raise ValueError('`timbre` must be provided along with `prompt_processed`!')
            timbre = timbre.unsqueeze(0).to(self.device)           
            prompts = prompt_processed.unsqueeze(0).to(self.device)

        batch_outputs = self.sample_batch(
            phonemes=phonemes,
            src_lens=torch.full((phonemes.size(0),), phonemes.size(-1), dtype=torch.long, device=self.device),
            prompts=prompts,
            timbres=timbre,
            codec_decoder=codec_decoder,
            temp_durgen=temp_durgen,
            temp_denoiser=temp_denoiser,
            nsteps_durgen=nsteps_durgen,
            nsteps_denoiser=nsteps_denoiser,
            guidance_scale=guidance_scale,
        )
        
        wav = batch_outputs['wav'][0][0].detach().cpu().numpy()
        
        end_time = time.time()
                
        return {
            'wav': wav,
            'time': end_time - start_time,
        }
    
    @torch.inference_mode()
    def sample_batch(
        self,
        phonemes: torch.Tensor,
        src_lens: torch.Tensor,
        prompts: torch.Tensor,
        timbres: torch.Tensor,
        codec_decoder: torch.nn.Module = None,
        temp_durgen: float = 0.3,
        temp_denoiser: float = 0.3,
        nsteps_durgen: int = 64,
        nsteps_denoiser: int = 64,
        guidance_scale: float | None = None,
    ):
        if self.prob_generator is None:
            raise ValueError('ProbGenerator is not initialized (pipeline excluded ProbGenerator); sampling is unavailable.')
        start_time = time.time()

        phonemes = phonemes.to(self.device)
        src_lens = src_lens.to(self.device)
        prompts = prompts.to(self.device)
        timbres = timbres.to(self.device)

        prior_emb_cond, prior_logits, tgt_mask = self.prior_generator.sample(
            texts=phonemes,
            src_lens=src_lens,
            max_src_len=phonemes.size(-1),
            prompts=prompts,
            prompts_len=prompts.size(-1),
            nfe=nsteps_durgen,
            temperature=temp_durgen,
        )

        latents = self.prob_generator.sample(
            cond=prior_emb_cond,
            spk=timbres,
            nfe=nsteps_denoiser,
            temperature=temp_denoiser,
            mask=~tgt_mask.unsqueeze(-1),
            guidance_scale=guidance_scale,
        )

        outputs = {
            'prior_embs': prior_emb_cond,
            'prior_logits': prior_logits,
            'tgt_mask': tgt_mask,
            'latents': latents,
            'time': time.time() - start_time,
        }
        
        if codec_decoder is not None:
            outputs['wav'] = codec_decoder.inference(latents, timbres)
        
        return outputs
    
    def _preprocess_acoustic_prompt(self, acoustic_prompt, sr=16000):
        if isinstance(acoustic_prompt, str):
            acoustic_prompt = librosa.load(acoustic_prompt, sr=sr)[0]
            acoustic_prompt = torch.from_numpy(acoustic_prompt).float()
            acoustic_prompt = acoustic_prompt.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(acoustic_prompt, np.ndarray):
            acoustic_prompt = torch.from_numpy(acoustic_prompt).float()
            acoustic_prompt = acoustic_prompt.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(acoustic_prompt, torch.Tensor):
            acoustic_prompt = acoustic_prompt.to(self.device)
        else:
            raise ValueError('Acoustic prompt must be one of [str, np.ndarray, torch.tensor]!')
        return acoustic_prompt
    
    def _get_codec_models(self, codec_cfg):
        codec_encoder = FACodecEncoder.from_pretrained(codec_cfg['encoder']).eval()
        codec_decoder = FACodecDecoder.from_pretrained(codec_cfg['decoder']).eval()
        return codec_encoder, codec_decoder
    
    def read_lexicon(self, lexicon_path=None):
        if not lexicon_path:
            lexicon_path = os.path.join(os.path.dirname(__file__), '..', 'lexicon', 'librispeech-lexicon.txt')
        lexicon = {}
        with open(lexicon_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon
    
    def _preprocess_english(
        self, 
        text, 
        lexicon_path=None, 
        cleaners=['english_cleaners']
        ):  
        text = text.rstrip(punctuation)
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in self.lexicon:
                phones += self.lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", self.g2p(w)))
        phones = "{sp " + " ".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")
        sequence = np.array(text_to_sequence(phones, cleaners))
        sequence = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        return sequence, text, phones

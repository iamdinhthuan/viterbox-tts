"""
Viterbox - Vietnamese Text-to-Speech
Based on Chatterbox architecture, fine-tuned for Vietnamese.
"""
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, List

from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from .models.t3 import T3, T3Config
from .models.t3.modules.cond_enc import T3Cond
from .models.s3gen import S3Gen, S3GEN_SR
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.voice_encoder import VoiceEncoder
from .models.tokenizers import MTLTokenizer


REPO_ID = "dolly-vn/viterbox"


def _split_text_to_sentences(text: str) -> List[str]:
    """Split text into sentences by punctuation marks."""
    # Split by . ? ! and keep the delimiter
    pattern = r'([.?!]+)'
    parts = re.split(pattern, text)
    
    sentences = []
    current = ""
    for i, part in enumerate(parts):
        if re.match(pattern, part):
            # This is punctuation, append to current sentence
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current = part
    
    # Don't forget remaining text without ending punctuation
    if current.strip():
        sentences.append(current.strip())
    
    return [s for s in sentences if s.strip()]


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Trim silence from audio."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def crossfade_concat(audios: List[np.ndarray], sr: int, fade_ms: int = 50, pause_ms: int = 500) -> np.ndarray:
    """
    Concatenate audio segments with crossfading and optional pause between sentences.
    
    Args:
        audios: List of audio arrays
        sr: Sample rate
        fade_ms: Crossfade duration in milliseconds
        pause_ms: Pause duration between sentences in milliseconds
    """
    if not audios:
        return np.array([])
    if len(audios) == 1:
        return audios[0]
    
    fade_samples = int(sr * fade_ms / 1000)
    pause_samples = int(sr * pause_ms / 1000)
    
    # Build result
    result = audios[0].copy()
    
    for i in range(1, len(audios)):
        next_audio = audios[i]
        
        # Add pause between sentences
        if pause_samples > 0:
            silence = np.zeros(pause_samples, dtype=result.dtype)
            result = np.concatenate([result, silence])
        
        if len(result) < fade_samples or len(next_audio) < fade_samples:
            # Too short for crossfade, just concatenate
            result = np.concatenate([result, next_audio])
            continue
        
        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        
        # Apply crossfade
        result_end = result[-fade_samples:] * fade_out
        next_start = next_audio[:fade_samples] * fade_in
        crossfaded = result_end + next_start
        
        # Combine
        result = np.concatenate([
            result[:-fade_samples],
            crossfaded,
            next_audio[fade_samples:]
        ])
    
    return result


@dataclass
class TTSConds:
    """Conditioning tensors for TTS generation"""
    t3: Union['T3Cond', dict]  # T3 conditioning (T3Cond object or dict)
    s3: dict  # S3Gen conditioning dict
    ref_wav: Optional[torch.Tensor] = None
    
    def save(self, path):
        def to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.cpu()
            elif isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            elif hasattr(x, '__dict__'):
                return {k: to_cpu(v) for k, v in vars(x).items()}
            return x
        
        torch.save({
            't3': to_cpu(self.t3),
            'gen': to_cpu(self.s3),
        }, path)
    
    @classmethod
    def load(cls, path, device):
        def to_device(x, dev):
            if isinstance(x, torch.Tensor):
                return x.to(dev)
            elif isinstance(x, dict):
                return {k: to_device(v, dev) for k, v in x.items()}
            return x
        
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle both old format (t3, s3) and new format (t3, gen)
        t3_data = data.get('t3', {})
        s3_data = data.get('gen', data.get('s3', {}))
        ref_wav = data.get('ref_wav', None)
        
        # Convert t3_data dict to T3Cond object
        if isinstance(t3_data, dict) and 'speaker_emb' in t3_data:
            t3_cond = T3Cond(
                speaker_emb=to_device(t3_data['speaker_emb'], device),
                cond_prompt_speech_tokens=to_device(t3_data.get('cond_prompt_speech_tokens'), device),
                cond_prompt_speech_emb=to_device(t3_data.get('cond_prompt_speech_emb'), device) if t3_data.get('cond_prompt_speech_emb') is not None else None,
                clap_emb=to_device(t3_data.get('clap_emb'), device) if t3_data.get('clap_emb') is not None else None,
                emotion_adv=to_device(t3_data.get('emotion_adv'), device) if t3_data.get('emotion_adv') is not None else None,
            )
        else:
            t3_cond = to_device(t3_data, device)
        
        return cls(
            t3=t3_cond,
            s3=to_device(s3_data, device),
            ref_wav=to_device(ref_wav, device) if ref_wav is not None else None,
        )


class Viterbox:
    """
    Vietnamese Text-to-Speech model.
    
    Example:
        >>> tts = Viterbox.from_pretrained("cuda")
        >>> audio = tts.generate("Xin chào!")
        >>> tts.save_audio(audio, "output.wav")
    """
    
    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str = "cuda",
    ):
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.sr = 24000  # Output sample rate
        self.conds: Optional[TTSConds] = None
        
    @classmethod
    def from_pretrained(cls, device: str = "cuda") -> 'Viterbox':
        """Load model from HuggingFace Hub"""
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt",
                    "t3_ml24ls_v2.safetensors",
                    "s3gen.pt",
                    "tokenizer_vi_expanded.json",
                    "conds.pt",
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    
    @classmethod
    def from_local(cls, ckpt_dir: Union[str, Path], device: str = "cuda") -> 'Viterbox':
        """Load model from local directory"""
        ckpt_dir = Path(ckpt_dir)
        
        # Load Voice Encoder
        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
        ve.to(device).eval()
        
        # Load T3 model
        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_ml24ls_v2.safetensors")
        
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        
        # Resize embeddings if needed
        if "text_emb.weight" in t3_state:
            old_emb = t3_state["text_emb.weight"]
            if old_emb.shape[0] != t3.hp.text_tokens_dict_size:
                new_emb = torch.zeros((t3.hp.text_tokens_dict_size, old_emb.shape[1]), dtype=old_emb.dtype)
                min_rows = min(old_emb.shape[0], new_emb.shape[0])
                new_emb[:min_rows] = old_emb[:min_rows]
                if new_emb.shape[0] > min_rows:
                    nn.init.normal_(new_emb[min_rows:], mean=0.0, std=0.02)
                t3_state["text_emb.weight"] = new_emb
        
        if "text_head.weight" in t3_state:
            old_head = t3_state["text_head.weight"]
            if old_head.shape[0] != t3.hp.text_tokens_dict_size:
                new_head = torch.zeros((t3.hp.text_tokens_dict_size, old_head.shape[1]), dtype=old_head.dtype)
                min_rows = min(old_head.shape[0], new_head.shape[0])
                new_head[:min_rows] = old_head[:min_rows]
                if new_head.shape[0] > min_rows:
                    nn.init.normal_(new_head[min_rows:], mean=0.0, std=0.02)
                t3_state["text_head.weight"] = new_head
        
        t3.load_state_dict(t3_state)
        t3.to(device).eval()
        
        # Load S3Gen
        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True))
        s3gen.to(device).eval()
        
        # Load tokenizer
        tokenizer = MTLTokenizer(str(ckpt_dir / "tokenizer_vi_expanded.json"))
        
        model = cls(t3, s3gen, ve, tokenizer, device)
        
        # Load default conditioning if exists
        conds_path = ckpt_dir / "conds.pt"
        if conds_path.exists():
            model.conds = TTSConds.load(conds_path, device)
        
        return model
    
    def prepare_conditionals(self, audio_prompt: Union[str, Path, torch.Tensor], exaggeration: float = 0.5):
        """
        Prepare conditioning from reference audio.
        
        Args:
            audio_prompt: Path to WAV file or audio tensor
            exaggeration: Expression intensity (0.0 - 2.0)
        """
        # Load audio at S3Gen sample rate (24kHz)
        if isinstance(audio_prompt, (str, Path)):
            s3gen_ref_wav, _ = librosa.load(str(audio_prompt), sr=S3GEN_SR, mono=True)
        else:
            s3gen_ref_wav = audio_prompt.cpu().numpy()
            if s3gen_ref_wav.ndim > 1:
                s3gen_ref_wav = s3gen_ref_wav.squeeze()
        
        # Resample to 16kHz for voice encoder and tokenizer
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        
        # Limit conditioning length
        DEC_COND_LEN = S3GEN_SR * 10  # 10 seconds max
        ENC_COND_LEN = S3_SR * 10
        s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
        
        with torch.inference_mode():
            # Get S3Gen conditioning
            s3_cond = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
            
            # Speech cond prompt tokens for T3
            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:ENC_COND_LEN]], max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)
            
            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)
            
            # Create T3Cond
            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)
        
        self.conds = TTSConds(t3=t3_cond, s3=s3_cond, ref_wav=torch.from_numpy(s3gen_ref_wav).unsqueeze(0))
        return self.conds
    
    def _generate_single(
        self,
        text: str,
        language: str,
        cfg_weight: float,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> np.ndarray:
        """Generate speech for a single sentence."""
        # Tokenize text with language prefix
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language).to(self.device)
        
        # Duplicate for CFG (classifier-free guidance needs two sequences)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        
        # Add start and stop tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device=='cuda')):
            # Generate speech tokens with T3
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
            )
            
            # Extract only the conditional batch and filter invalid tokens
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)
        
        # Generate waveform with S3Gen
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.s3,
        )
        
        return wav[0].cpu().numpy()
    
    def generate(
        self,
        text: str,
        language: str = "vi",
        audio_prompt: Optional[Union[str, Path, torch.Tensor]] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        split_sentences: bool = True,
        crossfade_ms: int = 50,
        sentence_pause_ms: int = 500,
    ) -> torch.Tensor:
        """
        Generate speech from text.
        
        Args:
            text: Input text to synthesize
            language: Language code ('vi' or 'en')
            audio_prompt: Optional reference audio for voice cloning
            exaggeration: Expression intensity (0.0 - 2.0)
            cfg_weight: Classifier-free guidance weight (0.0 - 1.0)
            temperature: Sampling temperature (0.1 - 1.0)
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty for T3
            split_sentences: Whether to split text by punctuation and generate separately
            crossfade_ms: Crossfade duration in milliseconds when merging sentences
            sentence_pause_ms: Pause duration between sentences in milliseconds (default 500ms)
            
        Returns:
            Audio tensor (1, samples) at 24kHz
        """
        # Prepare conditioning if reference provided
        if audio_prompt is not None:
            self.prepare_conditionals(audio_prompt, exaggeration)
        
        if self.conds is None:
            raise ValueError("No conditioning available. Provide audio_prompt or load model with default conds.")
        
        if split_sentences:
            # Split text into sentences
            sentences = _split_text_to_sentences(text)
            
            if len(sentences) == 0:
                sentences = [text]
            elif len(sentences) == 1:
                # Single sentence, no need for splitting logic
                pass
            
            # Generate each sentence
            audio_segments = []
            for i, sentence in enumerate(sentences):
                print(f"  [{i+1}/{len(sentences)}] {sentence[:50]}...")
                
                audio_np = self._generate_single(
                    text=sentence,
                    language=language,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                
                # Trim silence from each segment
                audio_np = trim_silence(audio_np, self.sr, top_db=30)
                
                if len(audio_np) > 0:
                    audio_segments.append(audio_np)
            
            # Merge with crossfading and pause
            if audio_segments:
                merged = crossfade_concat(audio_segments, self.sr, fade_ms=crossfade_ms, pause_ms=sentence_pause_ms)
                return torch.from_numpy(merged).unsqueeze(0)
            else:
                return torch.zeros(1, self.sr)  # 1 second of silence as fallback
        else:
            # Single generation without splitting
            audio_np = self._generate_single(
                text=text,
                language=language,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            return torch.from_numpy(audio_np).unsqueeze(0)
    
    def save_audio(self, audio: torch.Tensor, path: Union[str, Path], trim_silence: bool = True):
        """
        Save audio to file.
        
        Args:
            audio: Audio tensor from generate()
            path: Output file path
            trim_silence: Whether to trim trailing silence
        """
        import soundfile as sf
        
        audio_np = audio[0].cpu().numpy()
        
        if trim_silence:
            audio_np, _ = librosa.effects.trim(audio_np, top_db=30)
        
        sf.write(str(path), audio_np, self.sr)

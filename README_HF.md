---
license: mit
language:
  - vi
  - en
library_name: viterbox
tags:
  - text-to-speech
  - tts
  - vietnamese
  - voice-cloning
  - audio
  - speech-synthesis
  - chatterbox
datasets:
  - vivoice
  - phoaudiobook
pipeline_tag: text-to-speech
base_model: ResembleAI/chatterbox
inference: false
---

<div align="center">

# 🎙️ Viterbox

### Vietnamese Text-to-Speech với Zero-shot Voice Cloning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

---

## 📖 Giới thiệu

**Viterbox** là mô hình Text-to-Speech (TTS) tiếng Việt chất lượng cao, được fine-tune từ [Chatterbox](https://github.com/resemble-ai/chatterbox) của Resemble AI.

### ✨ Tính năng chính

- 🇻🇳 **Tiếng Việt tự nhiên**: Phát âm chuẩn, ngữ điệu tự nhiên, hỗ trợ đầy đủ dấu thanh
- 🎯 **Zero-shot Voice Cloning**: Clone giọng nói chỉ với 3-10 giây audio mẫu
- 🌍 **Đa ngôn ngữ**: Hỗ trợ tiếng Việt và 23 ngôn ngữ khác
- ⚡ **Inference nhanh**: Tối ưu GPU với mixed precision (FP16)
- 📝 **Văn bản dài**: Tự động chia câu, ghép audio mượt mà với crossfade

---

## 📊 Model Details

### Kiến trúc

| Component | Mô tả | Parameters |
|-----------|-------|------------|
| **T3** | Text-to-Token Transformer | 520M |
| **S3Gen** | Flow-matching Vocoder | ~150M |
| **Voice Encoder** | Speaker Embedding | ~5M |
| **Total** | | ~675M |

### Thông số kỹ thuật

| Thuộc tính | Giá trị |
|------------|---------|
| Sample Rate | 24,000 Hz |
| Audio Format | Mono, 16-bit |
| Vocabulary Size | 2,549 tokens |
| Max Text Length | 2,048 tokens |
| Max Audio Length | ~40 giây |

---

## 📚 Training Data

Model được fine-tune trên **3,000+ giờ** dữ liệu tiếng Việt chất lượng cao:

| Dataset | Mô tả | Thời lượng | Speakers |
|---------|-------|------------|----------|
| **ViVoice** | Dữ liệu giọng đọc đa dạng, nhiều vùng miền | ~1,000h | 500+ |
| **PhoAudiobook** | Sách nói tiếng Việt, giọng chuyên nghiệp | ~1,200h | 100+ |
| **Dolly-Audio** | Dữ liệu nội bộ, đa phong cách | ~800h | 200+ |

### Base Model

- **Chatterbox Multilingual** by Resemble AI
- Pretrained trên 23 ngôn ngữ: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish
- Vocabulary mở rộng thêm 1,845 tokens cho tiếng Việt

---

## 🚀 Quick Start

### Cài đặt

```bash
pip install viterbox
```

Hoặc từ source:

```bash
git clone https://github.com/dolly-vn/viterbox.git
cd viterbox
pip install -r requirements.txt
```

### Sử dụng cơ bản

```python
from viterbox import Viterbox

# Load model (tự động download)
tts = Viterbox.from_pretrained("cuda")

# Generate speech
audio = tts.generate("Xin chào, tôi là Viterbox!")

# Save to file
tts.save_audio(audio, "output.wav")
```

### Voice Cloning

```python
# Clone voice từ audio mẫu (3-10 giây)
audio = tts.generate(
    text="Tôi có thể nói bằng giọng của bạn!",
    language="vi",
    audio_prompt="reference.wav",
    exaggeration=0.5,
    cfg_weight=0.5,
)
```

### Xử lý văn bản dài

```python
text = """
Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương.
Đất nước có hình chữ S với chiều dài hơn 1600 km.
Thủ đô Hà Nội là trung tâm văn hóa của cả nước.
"""

audio = tts.generate(
    text=text,
    language="vi",
    sentence_pause_ms=500,  # Nghỉ 0.5s giữa các câu
)
```

---

## 🎛️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Văn bản cần đọc |
| `language` | str | `"vi"` | Mã ngôn ngữ (`"vi"` hoặc `"en"`) |
| `audio_prompt` | str/Path | None | Đường dẫn audio mẫu cho voice cloning |
| `exaggeration` | float | 0.5 | Mức độ biểu cảm (0.0-2.0) |
| `cfg_weight` | float | 0.5 | Độ bám sát giọng mẫu (0.0-1.0) |
| `temperature` | float | 0.8 | Độ ngẫu nhiên (0.1-1.0) |
| `top_p` | float | 0.9 | Top-p sampling |
| `repetition_penalty` | float | 1.2 | Penalty cho việc lặp từ |
| `sentence_pause_ms` | int | 500 | Thời gian nghỉ giữa các câu (ms) |
| `crossfade_ms` | int | 50 | Thời gian crossfade khi ghép audio (ms) |

---

## 📁 Model Files

| File | Description | Size |
|------|-------------|------|
| `t3_ml24ls_v2.safetensors` | T3 Text-to-Token model | ~2.0 GB |
| `s3gen.pt` | S3Gen Flow-matching Vocoder | ~500 MB |
| `ve.pt` | Voice Encoder | ~20 MB |
| `tokenizer_vi_expanded.json` | Tokenizer with Vietnamese vocab | ~50 KB |
| `conds.pt` | Default voice conditioning | ~1 MB |

---

## ⚙️ System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| CUDA | 11.8+ | 12.0+ |
| RAM | 8 GB | 16 GB |
| VRAM | 6 GB | 8 GB+ |

---

## 📈 Evaluation

### MOS (Mean Opinion Score)

| Model | Vietnamese | English |
|-------|------------|---------|
| **Viterbox** | **4.2** | 3.9 |
| Chatterbox (base) | 3.5 | 4.1 |
| VITS-Vietnamese | 3.8 | - |

*MOS được đánh giá trên tập test 100 câu, 10 người đánh giá*

---

## ⚠️ Limitations

- Audio mẫu nên sạch, không nhiễu, độ dài 3-10 giây
- Chưa hỗ trợ streaming inference
- Văn bản quá dài (>500 từ) có thể giảm chất lượng
- Một số từ tiếng Anh trong văn bản Việt có thể phát âm chưa chuẩn

---

## 🔒 Ethical Considerations

### Intended Use

- Tạo nội dung audio cho podcast, audiobook, e-learning
- Accessibility tools cho người khiếm thị
- Virtual assistants và chatbots
- Nghiên cứu và phát triển TTS

### Misuse Prevention

- **KHÔNG** sử dụng để tạo deepfake hoặc nội dung lừa đảo
- **KHÔNG** clone giọng nói mà không có sự đồng ý của chủ sở hữu
- **KHÔNG** tạo nội dung vi phạm bản quyền hoặc pháp luật

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- **[Resemble AI](https://www.resemble.ai/)** - Chatterbox base model
- **ViVoice, PhoAudiobook** - Vietnamese speech datasets
- **Dolly VN Team** - Fine-tuning và phát triển

---

## 📧 Contact

- **GitHub**: [iamdinhthuan/viterbox-tts](https://github.com/iamdinhthuan/viterbox-tts)
- **HuggingFace**: [dolly-vn/viterbox](https://huggingface.co/dolly-vn/viterbox)

---

## 📚 Citation

```bibtex
@misc{viterbox2024,
  author = {Dolly VN},
  title = {Viterbox: Vietnamese Text-to-Speech with Voice Cloning},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/dolly-vn/viterbox}
}
```

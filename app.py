"""
Viterbox - Gradio Web Interface
"""
import torch
import numpy as np
import librosa
from pathlib import Path
import warnings
import random
import gradio as gr

warnings.filterwarnings('ignore')

from viterbox import Viterbox

# Sample sentences
SAMPLES = {
    "vi": [
        "Xin chào các bạn! Tôi là Viterbox, trợ lý giọng nói thông minh được phát triển bởi đội ngũ Dolly VN. Rất vui được gặp các bạn hôm nay.",
        "Việt Nam là một quốc gia xinh đẹp nằm ở Đông Nam Á, với bề dày lịch sử hơn bốn nghìn năm văn hiến. Từ những ruộng bậc thang Sapa đến những bãi biển xanh ngắt Phú Quốc, đất nước hình chữ S luôn khiến du khách phải say đắm.",
        "Công nghệ trí tuệ nhân tạo đang phát triển với tốc độ chóng mặt. Chỉ trong vài năm gần đây, AI đã có thể viết văn, vẽ tranh, và giờ đây là tổng hợp giọng nói tự nhiên như con người.",
        "Hà Nội, thủ đô ngàn năm văn hiến của Việt Nam, nổi tiếng với ba mươi sáu phố phường và những món ăn đường phố tuyệt vời. Phở, bún chả, cà phê trứng là những đặc sản mà bạn không thể bỏ qua khi đến đây.",
        "Chào mừng bạn đến với thế giới của Text-to-Speech! Với Viterbox, bạn có thể biến bất kỳ văn bản nào thành giọng nói tự nhiên, mượt mà chỉ trong vài giây.",
    ],
    "en": [
        "Hello everyone! I am Viterbox, an intelligent voice assistant developed by the Dolly VN team. It's a pleasure to meet you today.",
        "Artificial intelligence technology is advancing at a breathtaking pace. In just a few years, AI has learned to write, create art, and now synthesize natural human-like speech.",
        "Vietnam is a beautiful country located in Southeast Asia, with over four thousand years of rich cultural heritage. From the terraced rice fields of Sapa to the pristine beaches of Phu Quoc, this S-shaped nation never fails to captivate visitors.",
        "Welcome to the world of Text-to-Speech technology! With Viterbox, you can transform any written text into natural, smooth voice output in just seconds.",
        "The future of human-computer interaction is voice. As we move towards a more connected world, voice assistants will become an integral part of our daily lives.",
    ],
}

# Load model
print("=" * 50)
print("🚀 Loading Viterbox...")
print("=" * 50)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

MODEL = Viterbox.from_pretrained(DEVICE)
print("✅ Model loaded!")
print("=" * 50)


def list_voices() -> list[str]:
    """List available voice files"""
    wav_dir = Path("wavs")
    if wav_dir.exists():
        return sorted([str(f) for f in wav_dir.glob("*.wav")])
    return []


def generate_speech(
    text: str,
    language: str,
    reference_audio,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    sentence_pause: float,
):
    """Generate speech from text"""
    if not text.strip():
        return None, "❌ Please enter some text"
    
    # Get reference audio path
    ref_path = reference_audio if reference_audio else None
    
    if ref_path is None and MODEL.conds is None:
        return None, "❌ Please provide reference audio"
    
    try:
        # Generate
        wav = MODEL.generate(
            text=text.strip(),
            language=language,
            audio_prompt=ref_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            sentence_pause_ms=int(sentence_pause * 1000),  # Convert seconds to ms
        )
        
        # Convert to numpy
        audio_np = wav[0].cpu().numpy()
        
        # Trim silence
        audio_np, _ = librosa.effects.trim(audio_np, top_db=30)
        
        duration = len(audio_np) / MODEL.sr
        status = f"✅ Generated! | {duration:.2f}s | {language.upper()}"
        
        return (MODEL.sr, audio_np), status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


# CSS
CSS = """
body, .gradio-container { background: #0f172a !important; }
.gradio-container { max-width: 100% !important; padding: 1rem 2rem !important; }
.status-badge { 
    display: inline-flex; align-items: center; padding: 4px 12px;
    border-radius: 999px; font-size: 0.8rem; font-weight: 500;
    background: #4f46e5; color: #fff;
}
#main-row { gap: 1rem !important; }
#main-row > div { flex: 1 !important; min-width: 0 !important; }
.card { 
    background: #1e293b !important; border-radius: 0.75rem;
    border: 1px solid #334155 !important; padding: 1rem 1.25rem; height: 100%;
}
.section-title { 
    font-size: 0.85rem; font-weight: 600; color: #e5e7eb;
    margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;
}
.generate-btn { 
    background: #4f46e5 !important; border-radius: 0.5rem !important;
    font-size: 1rem !important; padding: 10px 24px !important; margin-top: 0.75rem !important;
}
.output-card { 
    background: #1e293b !important; border-radius: 0.75rem;
    border: 1px solid #334155 !important; padding: 1rem 1.25rem; margin-top: 0.75rem;
}
"""

# Build UI
with gr.Blocks(
    title="🎙️ Viterbox TTS",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate"),
    css=CSS
) as demo:
    
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 0.5rem;">
            <h1 style="margin: 0; font-size: 2rem;">🎙️ Viterbox TTS</h1>
            <p style="color: #6b7280; margin-top: 0.5rem;">Vietnamese & English Text-to-Speech</p>
        </div>
    """)
    
    gr.HTML('<div style="text-align: center; margin-bottom: 1rem;"><span class="status-badge">🎯 Fine-tuned Model</span></div>')
    
    with gr.Row(equal_height=True, elem_id="main-row"):
        # Left - Text Input
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="section-title">📝 Text Input</div>')
            
            language = gr.Radio(
                choices=[("🇻🇳 Tiếng Việt", "vi"), ("🇺🇸 English", "en")],
                value="vi", label="Language"
            )
            
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Nhập văn bản cần đọc...",
                lines=5
            )
            
            with gr.Row():
                sample_btn = gr.Button("🎲 Random", variant="secondary", size="sm")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
        
        # Right - Voice & Settings
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="section-title">🎤 Reference Voice</div>')
            
            wav_files = list_voices()
            if wav_files:
                ref_dropdown = gr.Dropdown(
                    choices=[(Path(f).stem, f) for f in wav_files],
                    label="Select Voice",
                    value=wav_files[0] if wav_files else None,
                )
            else:
                ref_dropdown = gr.Dropdown(choices=[], label="No voices in wavs/")
            
            ref_audio = gr.Audio(label="Or Upload/Record", type="filepath", sources=["upload", "microphone"])
            
            gr.HTML('<div class="section-title" style="margin-top: 0.75rem;">⚙️ Settings</div>')
            
            exaggeration = gr.Slider(0.0, 2.0, 0.5, step=0.05, label="Exaggeration", info="Expression")
            
            with gr.Row():
                cfg_weight = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="CFG Weight", info="Voice adherence")
                temperature = gr.Slider(0.1, 1.0, 0.8, step=0.05, label="Temperature", info="Variation")
            
            sentence_pause = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Sentence Pause (s)", info="Pause between sentences")
    
    # Generate button
    generate_btn = gr.Button("🔊 Generate Speech", variant="primary", size="lg", elem_classes=["generate-btn"])
    
    # Output
    with gr.Column(elem_classes=["output-card"]):
        gr.HTML('<div class="section-title">🔈 Output</div>')
        with gr.Row():
            output_audio = gr.Audio(label="Generated Speech", type="numpy", scale=2)
            status_text = gr.Textbox(label="Status", lines=2, scale=1)
    
    # Handlers
    sample_btn.click(
        fn=lambda lang: random.choice(SAMPLES.get(lang, SAMPLES["vi"])),
        inputs=[language],
        outputs=[text_input]
    )
    clear_btn.click(fn=lambda: "", outputs=[text_input])
    ref_dropdown.change(fn=lambda x: x, inputs=[ref_dropdown], outputs=[ref_audio])
    
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, language, ref_audio, exaggeration, cfg_weight, temperature, sentence_pause],
        outputs=[output_audio, status_text]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

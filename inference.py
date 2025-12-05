"""
Viterbox - Command Line Inference
"""
import argparse
from pathlib import Path
from viterbox import Viterbox


def main():
    parser = argparse.ArgumentParser(description="Viterbox Text-to-Speech")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--lang", "-l", type=str, default="vi", help="Language (vi/en)")
    parser.add_argument("--ref", "-r", type=str, default=None, help="Reference audio for voice cloning")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output file path")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Expression intensity (0.0-2.0)")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight (0.0-1.0)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.1-1.0)")
    parser.add_argument("--sentence-pause", type=float, default=0.5, help="Pause between sentences in seconds (default 0.5)")
    
    args = parser.parse_args()
    
    print("Loading model...")
    tts = Viterbox.from_pretrained(args.device)
    print("✅ Model loaded")
    
    print(f"Generating: '{args.text}'")
    
    audio = tts.generate(
        text=args.text,
        language=args.lang,
        audio_prompt=args.ref,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        sentence_pause_ms=int(args.sentence_pause * 1000),
    )
    
    tts.save_audio(audio, args.output)
    print(f"✅ Saved to: {args.output}")


if __name__ == "__main__":
    main()

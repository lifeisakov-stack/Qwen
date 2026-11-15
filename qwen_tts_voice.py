#!/usr/bin/env python3
"""
Озвучка через Qwen3-TTS (локально, на видеокарте).

Установка (один раз):
    conda create -n qwen3-tts python=3.12 -y
    conda activate qwen3-tts
    pip install -U qwen-tts soundfile numpy
    # по желанию, быстрее и меньше VRAM:
    pip install -U flash-attn --no-build-isolation

Примеры:
    python qwen_tts_voice.py --text "Привет! Сегодня про инфляцию." --lang ru -o voice_ru.wav
    python qwen_tts_voice.py --file script_en.txt --lang en --speaker Ryan -o voice_en.wav
    python qwen_tts_voice.py --list-speakers

Текст в файле: одна фраза = одна строка. Пустые строки пропускаются.
Рядом с аудио создаётся .json с таймингами каждой строки (для субтитров).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODELS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",  # лучше качество
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",  # легче, быстрее
}
LANGS = {"ru": "Russian", "en": "English"}
DEFAULT_SPEAKER = {"ru": "Ryan", "en": "Ryan"}  # поменяй на любой из --list-speakers
GAP_SEC = 0.25  # пауза между строками


def load_model(size: str) -> Qwen3TTSModel:
    kwargs = dict(device_map="cuda:0", dtype=torch.bfloat16)
    try:
        return Qwen3TTSModel.from_pretrained(
            MODELS[size], attn_implementation="flash_attention_2", **kwargs
        )
    except Exception as e:  # нет flash-attn — работаем без него
        print(f"[!] flash_attention_2 недоступен ({e.__class__.__name__}), использую обычный режим")
        return Qwen3TTSModel.from_pretrained(MODELS[size], **kwargs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", help="текст прямо в команде")
    p.add_argument("--file", help="txt-файл, одна фраза на строку")
    p.add_argument("--lang", choices=LANGS, default="ru")
    p.add_argument("--speaker", help="имя голоса (см. --list-speakers)")
    p.add_argument("--instruct", default="", help='стиль, напр. "Calm, confident narrator."')
    p.add_argument("--model", choices=MODELS, default="1.7B")
    p.add_argument("-o", "--out", default="voice.wav")
    p.add_argument("--mp3", action="store_true", help="ещё и mp3 (нужен ffmpeg)")
    p.add_argument("--list-speakers", action="store_true")
    a = p.parse_args()

    model = load_model(a.model)

    if a.list_speakers:
        print("Голоса:", model.get_supported_speakers())
        print("Языки:", model.get_supported_languages())
        return

    if a.file:
        lines = Path(a.file).read_text(encoding="utf-8").splitlines()
    elif a.text:
        lines = [a.text]
    else:
        sys.exit("Нужен --text или --file")
    lines = [l.strip() for l in lines if l.strip()]

    speaker = a.speaker or DEFAULT_SPEAKER[a.lang]
    n = len(lines)
    wavs, sr = model.generate_custom_voice(
        text=lines,
        language=[LANGS[a.lang]] * n,
        speaker=[speaker] * n,
        instruct=[a.instruct] * n,
    )

    gap = np.zeros(int(sr * GAP_SEC), dtype=np.float32)
    parts, timings, t = [], [], 0.0
    for line, w in zip(lines, wavs):
        w = np.asarray(w, dtype=np.float32).squeeze()
        dur = len(w) / sr
        timings.append({"text": line, "start": round(t, 3), "end": round(t + dur, 3)})
        parts += [w, gap]
        t += dur + GAP_SEC

    out = Path(a.out)
    sf.write(out, np.concatenate(parts[:-1]), sr)
    out.with_suffix(".json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Готово: {out} ({t - GAP_SEC:.1f} сек), тайминги: {out.with_suffix('.json')}")

    if a.mp3:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out), "-q:a", "2", str(out.with_suffix(".mp3"))],
            check=True,
        )


if __name__ == "__main__":
    main()

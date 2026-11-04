"""Folha de frames de um Reel exportado, para o assistente ver antes de entregar.

Tira um frame a cada 0,5 s e junta-os numa grelha com o segundo de cada um.

Uso: python folha_frames.py <video.mp4> <pasta_saida>
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

FF = imageio_ffmpeg.get_ffmpeg_exe()


def main():
    video, saida = Path(sys.argv[1]), Path(sys.argv[2])
    saida.mkdir(parents=True, exist_ok=True)
    info = subprocess.run([FF, "-hide_banner", "-i", str(video)], capture_output=True, text=True).stderr
    h, m, s = info.split("Duration: ")[1].split(",")[0].split(":")
    duracao = int(h) * 3600 + int(m) * 60 + float(s)
    raw = subprocess.run([FF, "-v", "error", "-i", str(video), "-vf", "fps=2,scale=135:240", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 240, 135, 3)
    colunas = 10
    folha = Image.new("RGB", (135 * colunas, 240 * ((len(frames) + colunas - 1) // colunas)))
    for i, fr in enumerate(frames):
        im = Image.fromarray(fr)
        ImageDraw.Draw(im).text((3, 3), f"{i / 2:.1f}", fill="red")
        folha.paste(im, ((i % colunas) * 135, (i // colunas) * 240))
    destino = saida / (video.stem + "_frames.jpg")
    folha.save(destino, quality=85)
    print(f"{duracao:.1f}s, {len(frames)} frames: {destino}")


if __name__ == "__main__":
    main()

"""Folha de um ou mais bastidores, com um momento por segundo, para o assistente ver o que cada um conta.

Uso: python ver_bastidores.py <pasta_bastidores> <folha.jpg> <IMG_1.MOV> [<IMG_2.MOV> ...]
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from analisar import duracao_video
from reel import filtro_sdr

FF = imageio_ffmpeg.get_ffmpeg_exe()
LW, LH = 120, 213


def main():
    pasta, destino, nomes = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]), sys.argv[3:]
    linhas = []
    for nome in nomes:
        v = pasta / nome
        raw = subprocess.run([FF, "-v", "error", "-i", str(v), "-vf",
                              f"{filtro_sdr(v)}fps=1,scale={LW}:{LH}:force_original_aspect_ratio=increase,crop={LW}:{LH}",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        linhas.append((nome, duracao_video(v), np.frombuffer(raw, np.uint8).reshape(-1, LH, LW, 3)))
    colunas = max(len(q) for _, _, q in linhas)
    folha = Image.new("RGB", (colunas * LW, len(linhas) * (LH + 16)), "white")
    d = ImageDraw.Draw(folha)
    for i, (nome, dur, quadros) in enumerate(linhas):
        y = i * (LH + 16)
        for j, q in enumerate(quadros):
            im = Image.fromarray(q)
            ImageDraw.Draw(im).text((3, 3), f"{j}s", fill="red")
            folha.paste(im, (j * LW, y))
        d.text((3, y + LH + 2), f"{nome} {dur}s", fill="black")
    folha.save(destino, quality=80)
    print(destino)


if __name__ == "__main__":
    main()

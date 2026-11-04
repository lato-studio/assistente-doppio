"""Gera um vídeo curto (MP4 H.264, 1080x1350, 30 fps) a partir de slides, para pôr num carrossel ou nos Reels.
Funciona no Mac e no Windows (usa o ffmpeg do imageio-ffmpeg).

Uso:
    python3 video.py video.json

O video.json usa os mesmos layouts do montar.py, e mais o bastidor:

{
  "saida": "<pasta do casamento>/videos/stop-motion-damas.mp4",
  "fundo": "#FFFFFF",
  "segundos_por_imagem": 0.5,
  "repeticoes": 3,
  "slides": [
    {"layout": "mosaico", "margem_x": 200, "margem_y": 205, "linhas": [["a.jpg"]]},
    {"layout": "bastidor", "ficheiro": "IMG_1.MOV", "inicio": 2.0, "segundos": 1.5}
  ],
  "por_cima": {"titulo": "...", "citacao": "...", "assinatura": "Ana & Rui"}
}

- Stop motion: as mesmas margens em todas as imagens e só a foto muda (0,4 a 0,6 s cada).
- Ritmo: fotos a toda a página (`cheia`) a 0,4 a 0,6 s cada; pode abrir com um bastidor.
- Testemunho em vídeo: `por_cima` escreve o testemunho, fixo, por cima de todas as imagens.
- Slideshow sobre cor: fotos pequenas (`mosaico` com margens) sobre `"fundo": "da-foto"` ou uma cor, 15 a 30 s.
- Colagem: `{"layout": "camadas", "fundo_foto": ..., "fotos": [...], "segundos_por_camada": 0.6}`;
  as fotos entram por cima do fundo, uma de cada vez, e ficam.
Um slide pode ter "segundos" próprio para ficar mais tempo (por exemplo, a última imagem).
O vídeo sai sem som: a música escolhe-se no Instagram.
"""
import json
import os
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import montar as mc  # noqa: E402

FPS = 30
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def filtro_sdr(caminho):
    """Os vídeos do iPhone vêm em HDR (HLG). Sem conversão ficam lavados; com ela ficam como no telemóvel."""
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", caminho], capture_output=True, text=True).stderr
    if "arib-std-b67" in info or "smpte2084" in info:
        return ("zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
                "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,")
    return ""


def frames_do_bastidor(caminho, inicio, n):
    W, H = mc.W, mc.H
    cmd = [FFMPEG, "-v", "error", "-ss", f"{inicio:.3f}", "-i", caminho, "-frames:v", str(n),
           "-vf", f"{filtro_sdr(caminho)}fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}",
           "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    if len(frames) == 0:
        sys.exit(f"Não consegui ler o bastidor {os.path.basename(caminho)} a partir de {inicio}s.")
    if len(frames) < n:
        frames = np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), 0)])
    return frames


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as fh:
        plano = json.load(fh)
    # um caminho relativo conta a partir da pasta do plano
    saida = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), os.path.expanduser(plano["saida"])))
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    mc.configurar(plano)
    padrao = plano.get("segundos_por_imagem", 0.5)
    por_cima = plano.get("por_cima")
    camada, zonas = mc.camada_de_texto(por_cima, plano) if por_cima else (None, ())

    # cada pedaço é uma imagem parada (imagem, n.º de frames) ou os frames de um bastidor
    pedacos = []
    for i, s in enumerate(plano["slides"], 1):
        n = max(1, round(s.get("segundos", padrao) * FPS))
        if s.get("layout") == "camadas":
            n_camada = max(1, round(s.get("segundos_por_camada", 0.6) * FPS))
            for im in mc.camadas_etapas(s, plano, i):
                pedacos.append((np.asarray(im.convert("RGB")), n_camada))
            continue
        if s.get("layout") == "bastidor":
            frames = frames_do_bastidor(os.path.expanduser(s["ficheiro"]), s.get("inicio", 0), n)
            if por_cima:
                frames = np.stack([np.asarray(mc.texto_por_cima(Image.fromarray(f), por_cima, plano, camada))
                                   for f in frames])
            pedacos.append(frames)
            continue
        # com texto por cima, o corte de cada foto guarda as caras fora das zonas do texto
        for im in mc.LAYOUTS[s["layout"]]({**s, "_zonas": zonas}, plano, i):
            if por_cima:
                im = mc.texto_por_cima(im, por_cima, plano, camada)
            pedacos.append((np.asarray(im.convert("RGB")), n))

    escritor = imageio_ffmpeg.write_frames(
        saida, (mc.W, mc.H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-crf", "16", "-preset", "slow", "-profile:v", "high", "-movflags", "+faststart",
                       "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"])
    escritor.send(None)
    total = 0
    for _ in range(plano.get("repeticoes", 1)):
        for p in pedacos:
            if isinstance(p, tuple):
                frame, n = p
                frame = np.ascontiguousarray(frame)
                for _ in range(n):
                    escritor.send(frame)
                total += n
            else:
                for f in p:
                    escritor.send(np.ascontiguousarray(f))
                total += len(p)
    escritor.close()
    print(f"{saida}: {total / FPS:.1f} s, {mc.W}x{mc.H}, {FPS} fps")
    if mc.AVISOS:
        print("\nRecortes a confirmar (pés, cabeças, mãos):")
        for a in mc.AVISOS:
            print("  " + a)


if __name__ == "__main__":
    main()

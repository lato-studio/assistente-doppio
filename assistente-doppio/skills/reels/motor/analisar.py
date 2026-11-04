"""Analisa uma pasta de fotos (e de bastidores) para o assistente planear um Reel.

Para cada foto: a orientação, a hora em que foi tirada (para a ordem do dia)
e o ponto de interesse para o recorte 9:16 (as caras, se as houver). Grava
fotos.json e uma folha de contactos numerada, que o assistente vê antes de
escolher o gancho, a ordem e o herói. Os bastidores aparecem com a duração, a
hora e as fotos tiradas nos mesmos minutos (o par bastidor e resultado).

Uso: python analisar.py <pasta_fotos> <pasta_trabalho> [<pasta_bastidores>]
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pillow_heif
from PIL import Image, ImageDraw, ImageOps

pillow_heif.register_heif_opener()
FOTOS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
VIDEOS = {".mp4", ".mov", ".m4v"}
# O detetor de caras clássico só existe no OpenCV 4. Sem ele, o foco fica neutro e o assistente acerta-o ao ver a folha.
CARAS = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") if hasattr(cv2, "CascadeClassifier") else None


def hora(im, ficheiro):
    exif = im.getexif()
    valor = exif.get_ifd(0x8769).get(36867) or exif.get(306)
    if valor:
        try:
            return datetime.strptime(str(valor), "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            pass
    return datetime.fromtimestamp(ficheiro.stat().st_mtime).isoformat()


def ponto_de_interesse(im):
    """(x, y) de 0 a 1: o centro das caras encontradas, ou um ponto neutro um pouco acima do meio."""
    if CARAS is None:
        return [0.5, 0.42], None
    pequeno = im.convert("L")
    escala = 800 / max(pequeno.size)
    pequeno = pequeno.resize((round(pequeno.width * escala), round(pequeno.height * escala)))
    cinza = np.array(pequeno)
    caras = CARAS.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24))
    if len(caras) == 0:
        return [0.5, 0.42], 0
    areas = np.array([w * h for (_, _, w, h) in caras], float)
    cx = sum((x + w / 2) * a for (x, _, w, _), a in zip(caras, areas)) / areas.sum()
    cy = sum((y + h / 2) * a for (_, y, _, h), a in zip(caras, areas)) / areas.sum()
    return [round(cx / cinza.shape[1], 3), round(cy / cinza.shape[0], 3)], len(caras)


def hora_video(caminho):
    """A hora local em que o vídeo foi gravado. O iPhone guarda-a em UTC, por isso converte-se para a hora do computador."""
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(caminho)],
                          capture_output=True, text=True).stderr
    for chave in ("com.apple.quicktime.creationdate", "creation_time"):
        for linha in info.splitlines():
            if linha.strip().startswith(chave):
                valor = linha.split(":", 1)[1].strip().replace("Z", "+00:00")
                try:
                    return datetime.fromisoformat(valor).astimezone().replace(tzinfo=None).isoformat(timespec="seconds")
                except ValueError:
                    pass
    return datetime.fromtimestamp(caminho.stat().st_mtime).isoformat(timespec="seconds")


def duracao_video(caminho):
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(caminho)],
                          capture_output=True, text=True).stderr
    try:
        h, m, s = info.split("Duration: ")[1].split(",")[0].split(":")
        return round(int(h) * 3600 + int(m) * 60 + float(s), 1)
    except (IndexError, ValueError):
        return None


def main():
    pasta, trabalho = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    bastidores = Path(sys.argv[3]).expanduser() if len(sys.argv) > 3 else None
    trabalho.mkdir(parents=True, exist_ok=True)

    fotos = []
    for f in sorted(p for p in pasta.iterdir() if p.suffix.lower() in FOTOS):
        im = ImageOps.exif_transpose(Image.open(f))
        w, h = im.size
        foco, n_caras = ponto_de_interesse(im)
        fotos.append({"ficheiro": f.name, "orientacao": "vertical" if h > w * 1.05 else "horizontal" if w > h * 1.05 else "quadrada",
                      "hora": hora(im, f), "foco": foco, "caras": n_caras, "_im": im})
    fotos.sort(key=lambda x: x["hora"])
    for n, f in enumerate(fotos, 1):
        f["n"] = n

    # folha de contactos numerada, pela ordem do dia
    lado, colunas = 220, 8
    linhas = (len(fotos) + colunas - 1) // colunas
    folha = Image.new("RGB", (colunas * lado, linhas * (lado + 18)), "white")
    d = ImageDraw.Draw(folha)
    for i, f in enumerate(fotos):
        t = f.pop("_im").copy()
        t.thumbnail((lado - 6, lado - 6))
        x, y = (i % colunas) * lado, (i // colunas) * (lado + 18)
        folha.paste(t.convert("RGB"), (x + (lado - t.width) // 2, y + (lado - t.height) // 2))
        d.text((x + 4, y + lado), f"{f['n']} {f['orientacao'][0].upper()}", fill="black")
    folha.save(trabalho / "folha_fotos.jpg", quality=82)

    clips = []
    if bastidores and bastidores.exists():
        from reel import filtro_sdr
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        miniaturas = []
        for v in sorted(p for p in bastidores.iterdir() if p.suffix.lower() in VIDEOS):
            dur = duracao_video(v) or 0
            filtro = filtro_sdr(v)
            hv = hora_video(v)
            # as fotos tiradas nos mesmos minutos: o "resultado" que este bastidor pode revelar
            perto = [f["n"] for f in fotos
                     if abs((datetime.fromisoformat(f["hora"]) - datetime.fromisoformat(hv)).total_seconds()) <= 240]
            clips.append({"n": len(clips) + 1, "ficheiro": v.name, "duracao_s": dur, "hdr": bool(filtro),
                          "hora": hv, "fotos_do_momento": perto})
            # três momentos de cada bastidor (a 20%, 50% e 80%), para o assistente escolher o trecho
            quadros = []
            for frac in (0.2, 0.5, 0.8):
                raw = subprocess.run([ff, "-v", "error", "-ss", f"{dur * frac:.2f}", "-i", str(v), "-frames:v", "1",
                                      "-vf", f"{filtro}scale=160:284:force_original_aspect_ratio=increase,crop=160:284",
                                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
                if len(raw) == 160 * 284 * 3:
                    quadros.append(Image.fromarray(np.frombuffer(raw, np.uint8).reshape(284, 160, 3)))
            miniaturas.append((clips[-1], quadros))
        if miniaturas:
            colunas = 4
            linhas = (len(miniaturas) + colunas - 1) // colunas
            folha_b = Image.new("RGB", (colunas * 500, linhas * 310), "white")
            db = ImageDraw.Draw(folha_b)
            for i, (c, quadros) in enumerate(miniaturas):
                x, y = (i % colunas) * 500, (i // colunas) * 310
                for j, q in enumerate(quadros):
                    folha_b.paste(q, (x + j * 163, y))
                db.text((x + 2, y + 288), f"B{c['n']} {c['ficheiro']} {c['duracao_s']}s (20/50/80%)", fill="black")
            folha_b.save(trabalho / "folha_bastidores.jpg", quality=80)

    (trabalho / "fotos.json").write_text(json.dumps({"pasta_fotos": str(pasta), "fotos": fotos,
                                                     "pasta_bastidores": str(bastidores) if bastidores else None,
                                                     "bastidores": clips}, ensure_ascii=False, indent=1))
    v = sum(f["orientacao"] == "vertical" for f in fotos)
    print(f"{len(fotos)} fotos ({v} verticais, {len(fotos) - v} horizontais ou quadradas), {len(clips)} bastidores")
    print(f"folha: {trabalho / 'folha_fotos.jpg'}")
    for c in clips:
        par = f"fotos {c['fotos_do_momento'][0]} a {c['fotos_do_momento'][-1]}" if c["fotos_do_momento"] else "sem fotos do momento"
        print(f"  B{c['n']} {c['ficheiro']} {c['hora'][11:16]} {c['duracao_s']}s: {par}")


if __name__ == "__main__":
    main()

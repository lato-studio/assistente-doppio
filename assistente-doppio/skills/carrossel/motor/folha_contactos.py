"""Cria folhas de contactos numeradas para escolher fotos sem abrir uma a uma.

Uso:
    python3 folha_contactos.py <pasta_fotos> <pasta_saida> [--por-folha 30] [--colunas 6]

Gera folha_01.jpg, folha_02.jpg, ... e indice.tsv (numero, ficheiro, orientacao, largura, altura, hora).
As fotos vão pela ordem do dia (a hora da foto), e pelo nome quando não há hora.
Os números nas miniaturas correspondem às linhas do indice.tsv.
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imagens import abrir, letra, listar_fotos  # noqa: E402

CELULA = 300


def hora(caminho):
    """A hora em que a foto foi tirada (EXIF), para pôr as fotos pela ordem do dia."""
    try:
        exif = Image.open(caminho).getexif()
        return exif.get_ifd(0x8769).get(36867) or exif.get(306) or ""
    except Exception:
        return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pasta")
    p.add_argument("saida")
    p.add_argument("--por-folha", type=int, default=30)
    p.add_argument("--colunas", type=int, default=6)
    a = p.parse_args()

    fotos = listar_fotos(a.pasta)
    if not fotos:
        sys.exit(f"Nenhuma foto encontrada em {a.pasta}")
    horas = {c: hora(c) for c in fotos}
    if all(horas.values()):
        fotos.sort(key=lambda c: (horas[c], c))
    os.makedirs(a.saida, exist_ok=True)
    f = letra("texto", 26)

    linhas_indice = ["numero\tficheiro\torientacao\tlargura\taltura\thora"]
    for inicio in range(0, len(fotos), a.por_folha):
        lote = fotos[inicio:inicio + a.por_folha]
        linhas = (len(lote) + a.colunas - 1) // a.colunas
        folha = Image.new("RGB", (a.colunas * CELULA, linhas * CELULA), "white")
        d = ImageDraw.Draw(folha)
        for i, caminho in enumerate(lote):
            n = inicio + i + 1
            try:
                im = abrir(caminho, tamanho_max=CELULA * 2)
            except Exception as e:  # ficheiro corrompido ou formato estranho
                print(f"Aviso: não consegui abrir {caminho}: {e}", file=sys.stderr)
                continue
            w, h = im.size
            orient = "vertical" if h > w * 1.05 else "horizontal" if w > h * 1.05 else "quadrada"
            linhas_indice.append(f"{n}\t{caminho}\t{orient}\t{w}\t{h}\t{horas[caminho]}")
            im.thumbnail((CELULA - 8, CELULA - 8))
            x = (i % a.colunas) * CELULA + (CELULA - im.width) // 2
            y = (i // a.colunas) * CELULA + (CELULA - im.height) // 2
            folha.paste(im, (x, y))
            bx, by = (i % a.colunas) * CELULA + 6, (i // a.colunas) * CELULA + 6
            d.rectangle((bx, by, bx + 58, by + 34), fill="black")
            d.text((bx + 6, by + 3), str(n), fill="white", font=f)
        nome = os.path.join(a.saida, f"folha_{inicio // a.por_folha + 1:02d}.jpg")
        folha.save(nome, quality=85)
        print(nome)

    with open(os.path.join(a.saida, "indice.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas_indice) + "\n")
    print(f"{len(fotos)} fotos, índice em {os.path.join(a.saida, 'indice.tsv')}")


if __name__ == "__main__":
    main()

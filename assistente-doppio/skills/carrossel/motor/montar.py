"""Monta o banco de slides de um casamento (e os carrosséis tirados dele) a partir de um plano JSON.

Uso:
    python3 montar.py plano.json              todos os slides do banco e os posts
    python3 montar.py plano.json --slides 7,12   só estes slides do banco (para afinar)

O plano está descrito em references/motor.md. Na pasta `saida` ficam:
    banco/NN-nome.jpg        os slides, pela ordem do plano (4:5, 1080x1350, sRGB, qualidade 95)
    posts/<nome>/            os carrosséis que o aluno pedir, montados com slides do banco (só se o plano os tiver)

E, na pasta do plano (para o assistente conferir, nunca para entregar):
    _previsualizacao.jpg     todos os slides numerados, com as etiquetas
    _banco.csv               número, ficheiro, etiquetas e fotos de cada slide

Para cada foto recortada escreve quanto se cortou em cima/baixo e dos lados,
para confirmares que não se cortaram pés, cabeças ou mãos.
"""
import csv
import json
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imagens import SRGB_BYTES, abrir, caber, letra, preencher  # noqa: E402
import recorte  # noqa: E402

FORMATOS = {"4:5": (1080, 1350), "3:4": (1080, 1440)}
W, H = FORMATOS["4:5"]
FOCO_PADRAO = (0.5, 0.45)
AVISOS = []
CORTES = []  # (slide, foto, miniatura, corte relativo, avisos) para a folha de cortes
LETRAS_DO_PLANO = {}


# ---------- utilitários ----------

def foto(spec):
    """Aceita "caminho.jpg" ou {"ficheiro": ..., "foco": [x, y], "zoom": 1.4, "pb": true}.

    As cores ficam como o fotógrafo as editou: aqui só se recorta e redimensiona.
    `foco` só é preciso para pedir um corte diferente do automático (que guarda as caras,
    as cabeças, os pés e o detalhe da foto).
    `zoom` aproxima à volta do foco (recorte editorial mais apertado).
    `pb` faz uma VARIANTE a preto e branco (só luminância, sem mexer em contraste):
    é sempre uma alternativa à versão a cores, nunca a substitui."""
    if isinstance(spec, str):
        spec = {"ficheiro": spec}
    caminho = os.path.expanduser(spec["ficheiro"])
    im = abrir(caminho)
    foco = tuple(spec["foco"]) if "foco" in spec else None
    zoom = spec.get("zoom", 1)
    if zoom > 1:
        foco = foco or FOCO_PADRAO
        fx, fy = im.width * foco[0], im.height * foco[1]
        w, h = round(im.width / zoom), round(im.height / zoom)
        x = min(max(round(fx - w / 2), 0), im.width - w)
        y = min(max(round(fy - h / 2), 0), im.height - h)
        im = im.crop((x, y, x + w, y + h))
        foco = ((fx - x) / w, (fy - y) / h)
    if spec.get("pb"):
        im = im.convert("L").convert("RGB")
    return im, foco, os.path.basename(caminho)


_CORES = {}


def cor_da_foto(spec):
    """Uma cor suave tirada da própria foto (o verde do jardim, a pedra, a terracota), para o fundo
    combinar com ela. A foto não muda: a cor só pinta o fundo à volta."""
    import colorsys
    chave = json.dumps(spec, sort_keys=True)
    if chave in _CORES:
        return _CORES[chave]
    im, _, _ = foto(spec)
    pequena = im.resize((60, 75)).quantize(8, method=Image.Quantize.MEDIANCUT)
    paleta = pequena.getpalette()
    melhor = None
    for n, i in pequena.getcolors():
        r, g, b = (v / 255 for v in paleta[3 * i:3 * i + 3])
        h, sat, v = colorsys.rgb_to_hsv(r, g, b)
        if v < 0.15 or v > 0.93:
            continue
        nota = n * (0.3 + sat)
        if melhor is None or nota > melhor[0]:
            melhor = (nota, h, sat, v)
    if melhor is None:
        _CORES[chave] = "#EDEAE4"
        return _CORES[chave]
    _, h, sat, v = melhor
    r, g, b = colorsys.hsv_to_rgb(h, min(sat, 0.32) * 0.85, min(max(v, 0.42), 0.62))
    _CORES[chave] = "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))
    return _CORES[chave]


def cor_de_fundo(slide, plano):
    """"fundo": uma cor ("#7A9170"), ou "da-foto" (a cor da primeira foto do slide),
    ou {"da_foto": "caminho.jpg"} (a cor de outra foto, para vários slides com o mesmo fundo)."""
    f = slide.get("fundo", plano.get("fundo", "#FFFFFF"))
    if f == "da-foto":
        specs = slide.get("fotos") or [c for linha in slide.get("linhas", []) for c in linha]
        return cor_da_foto(specs[0])
    if isinstance(f, dict):
        return cor_da_foto(f["da_foto"])
    return f


def tela(slide, plano):
    return Image.new("RGB", (W, H), cor_de_fundo(slide, plano))


def pelicula(t, caixa, semente=0):
    """Desenha por cima da foto, na caixa, a borda preta e irregular de um negativo digitalizado.
    A borda tapa só os primeiros 2 a 3% da foto à volta; a cor da foto não muda."""
    x, y, w, h = caixa
    b = max(8, round(min(w, h) * 0.028))
    rnd = np.random.default_rng(semente)
    janela = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(janela)
    d.rounded_rectangle((b, b, w - b, h - b), radius=round(b * 0.7), fill=255)
    janela = janela.filter(ImageFilter.GaussianBlur(b * 0.18))
    ruido = rnd.uniform(-70, 70, (h, w)).astype(np.float32)
    ruido = np.asarray(Image.fromarray(ruido + 128).convert("L").filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) - 128
    jan = (np.asarray(janela).astype(np.float32) + ruido) > 128
    # o bordo de fora também é irregular, como a película recortada
    fora = Image.new("L", (w, h), 0)
    ImageDraw.Draw(fora).rectangle((1, 1, w - 2, h - 2), fill=255)
    fora = (np.asarray(fora.filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32) + ruido * 0.8) > 128
    moldura = Image.fromarray(((~jan) & fora).astype(np.uint8) * 255)
    fundo = t.crop((x, y, x + w, y + h))
    fundo.paste((14, 13, 12), (0, 0), moldura)
    resto = Image.fromarray((~fora).astype(np.uint8) * 255)
    fundo.paste(t.getpixel((0, 0)), (0, 0), resto)  # fora da película fica o fundo do slide
    t.paste(fundo, (x, y))


def colar_preenchido(t, spec, caixa, slide_n, zonas=()):
    """Cola a foto a encher a caixa. O corte é escolhido pelo recorte.py, e o que não deu para
    salvar (uma cara cortada, os pés à justa, o texto por cima de uma cara) fica nos avisos.

    zonas: as faixas (y0, y1), em px do slide, onde vai haver texto."""
    x, y, w, h = caixa
    im, foco, nome = foto(spec)
    zonas_caixa = [((z0 - y) / h, (z1 - y) / h) for z0, z1 in zonas]
    pedaco, corte, avisos = recorte.recortar(im, w, h, foco, zonas_caixa)
    perdido_v = 1 - (corte[3] - corte[1]) / im.height
    perdido_h = 1 - (corte[2] - corte[0]) / im.width
    if perdido_v > 0.01:
        onde = f"corta {perdido_v:.0%} na altura ({corte[1] / im.height:.0%} em cima)"
    elif perdido_h > 0.01:
        onde = f"corta {perdido_h:.0%} na largura ({corte[0] / im.width:.0%} à esquerda)"
    else:
        onde = ""
    if perdido_v > 0.03 or perdido_h > 0.03:
        mini = im.copy()
        mini.thumbnail((300, 300))
        CORTES.append((slide_n, nome, mini, (corte[0] / im.width, corte[1] / im.height,
                                             corte[2] / im.width, corte[3] / im.height), avisos))
    if avisos or perdido_v > 0.2 or perdido_h > 0.2:
        AVISOS.append(f"slide {slide_n} · {nome}: " + "; ".join(([onde] if onde else []) + avisos))
    t.paste(pedaco, (x, y))
    if isinstance(spec, dict) and spec.get("pelicula"):
        pelicula(t, caixa, semente=len(nome) + x + y)


def dividir(total, pesos, espaco):
    """Divide um comprimento em partes proporcionais aos pesos, com espaço entre elas."""
    util = total - espaco * (len(pesos) - 1)
    soma = sum(pesos)
    tamanhos = [round(util * p / soma) for p in pesos]
    tamanhos[-1] = util - sum(tamanhos[:-1])
    pos, out = 0, []
    for t in tamanhos:
        out.append((pos, t))
        pos += t + espaco
    return out


# ---------- texto (testemunho e frase) ----------

def _palavras_ricas(texto):
    """"Só as *fotografias* ficam." -> [[("Só", False)], [("as", False)], [("fotografias", True)], [("ficam.", False)]]

    Cada palavra é uma lista de pedaços (texto, itálico), para "*vosso*." ficar colado ao ponto.
    Uma mudança de linha no texto ("\n") força a quebra e fica como None."""
    palavras, italico, colar = [], False, False
    for parte in texto.split("*"):
        for tok in re.split(r"(\s+)", parte):
            if not tok:
                continue
            if tok.isspace():
                if "\n" in tok:
                    palavras.append(None)
                colar = False
                continue
            if colar and palavras and palavras[-1] is not None:
                palavras[-1].append((tok, italico))
            else:
                palavras.append([(tok, italico)])
            colar = True
        italico = not italico
    return palavras


def _quebrar(palavras, largura_max, medir):
    linhas, atual = [], []
    for p in palavras:
        if atual and medir(atual + [p]) > largura_max:
            linhas.append(atual)
            atual = []
        atual.append(p)
    return linhas + ([atual] if atual else [])


def _equilibrar(palavras, largura_max, medir):
    """Quebra em linhas e depois aperta a largura até onde der sem acrescentar linhas (o text-wrap: balance).
    Cada parágrafo (separado por None) equilibra-se à parte."""
    paragrafos, atual = [], []
    for p in palavras + [None]:
        if p is None:
            if atual:
                paragrafos.append(atual)
            atual = []
        else:
            atual.append(p)
    linhas = []
    for par in paragrafos:
        n = len(_quebrar(par, largura_max, medir))
        baixo, alto = largura_max * 0.3, largura_max
        while alto - baixo > 4:
            meio = (baixo + alto) / 2
            if len(_quebrar(par, meio, medir)) <= n and all(medir([p]) <= meio for p in par):
                alto = meio
            else:
                baixo = meio
        linhas += _quebrar(par, alto, medir)
    return linhas


def _medidor(d, fn, espaco):
    def medir(ps):
        return sum(d.textlength(t, font=fn[i]) for p in ps for t, i in p) + espaco * (len(ps) - 1)
    return medir


def _escrever_linha(d, linha, y, fn, espaco, medir, cor):
    x = (W - medir(linha)) / 2
    for p in linha:
        for t, i in p:
            d.text((x, y), t, font=fn[i], fill=cor)
            x += d.textlength(t, font=fn[i])
        x += espaco


def _sombra(camada, raio, forca):
    alfa = camada.getchannel("A").filter(ImageFilter.GaussianBlur(raio)).point(lambda v: int(v * forca))
    sombra = Image.new("RGBA", camada.size, (0, 0, 0, 0))
    sombra.putalpha(alfa)
    return Image.alpha_composite(sombra, camada)


def _veu(altura, paragens, de_baixo=False):
    """Um degradê preto: paragens = [(posição 0-1, opacidade 0-1), ...] a partir da borda."""
    v = Image.new("L", (1, altura))
    for y in range(altura):
        t = y / max(1, altura - 1)
        for (t0, a0), (t1, a1) in zip(paragens, paragens[1:]):
            if t0 <= t <= t1:
                v.putpixel((0, altura - 1 - y if de_baixo else y), round(255 * (a0 + (a1 - a0) * (t - t0) / (t1 - t0))))
                break
    camada = Image.new("RGBA", (W, altura), (0, 0, 0, 255))
    camada.putalpha(v.resize((W, altura)))
    return camada


def _estrela(d, cx, cy, r, cor):
    import math
    pontos = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pontos.append((cx + rr * math.cos(a), cy - rr * math.sin(a)))
    d.polygon(pontos, fill=cor)


def camada_de_texto(s, plano=None):
    """O texto por cima da foto, como o post de testemunho do Sérgio, numa camada transparente:
    - "titulo": a frase grande em serifada, em cima; *assim* fica em itálico; até 3 linhas;
    - "citacao": o texto do cliente, em baixo, até 3 linhas;
    - "estrelas": 5 (0 para nenhuma); "assinatura": o casal, em maiúsculas espaçadas;
    - "aspas": as aspas à volta do título (por defeito, sim quando há assinatura);
    - "titulo_y": onde começa o título (150).

    Devolve (camada, zonas): as zonas são as faixas (y0, y1), em px, onde há texto. As caras não podem lá ficar."""
    plano = plano or {}
    letras = {**LETRAS_DO_PLANO, **plano.get("letras", {})}
    titulo, citacao, nome = s.get("titulo"), s.get("citacao"), s.get("assinatura")
    camada = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(camada)
    branco = s.get("cor_texto", "#FFFFFF")

    if titulo:
        if s.get("aspas", bool(nome)):
            titulo = "“" + titulo.strip() + "”"
        palavras = _palavras_ricas(titulo)
        tamanho = s.get("tamanho_titulo", 92)
        while True:
            fn = {False: letra("titulo", tamanho, letras.get("titulo")),
                  True: letra("titulo_italico", tamanho, letras.get("titulo_italico"))}
            espaco = d.textlength(" ", font=fn[False])
            medir = _medidor(d, fn, espaco)
            linhas = _equilibrar(palavras, s.get("largura_titulo", 720), medir)
            if len(linhas) <= 3 or tamanho <= 56:
                break
            tamanho -= 2
        y = s.get("titulo_y", 150)
        for linha in linhas:
            _escrever_linha(d, linha, y, fn, espaco, medir, branco)
            y += tamanho * 1.02
        camada = _sombra(camada, 24, 0.3)
        d = ImageDraw.Draw(camada)

    blocos = []  # de baixo para cima: nome, citação, estrelas
    if nome:
        fnome = letra("texto", 17, letras.get("texto"))
        blocos.append(("nome", fnome, 17))
    if citacao:
        tam = s.get("tamanho_citacao", 30)
        while True:
            fc = {False: letra("texto", tam, letras.get("texto")), True: letra("texto", tam, letras.get("texto"))}
            esp = d.textlength(" ", font=fc[False])
            medir_c = _medidor(d, fc, esp)
            linhas_c = _equilibrar(_palavras_ricas(citacao), W - 200, medir_c)
            if len(linhas_c) <= 3 or tam <= 24:
                break
            tam -= 2
        blocos.append(("citacao", (fc, esp, linhas_c, medir_c), tam))
    estrelas = s.get("estrelas", 5 if (citacao or nome) else 0)

    y = H - s.get("margem_baixo", 80)
    texto_baixo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    db = ImageDraw.Draw(texto_baixo)
    for tipo, f, tam in blocos:
        if tipo == "nome":
            letras_nome = nome.upper()
            passo = tam * 0.3
            largura = sum(db.textlength(c, font=f) for c in letras_nome) + passo * (len(letras_nome) - 1)
            y -= tam * 1.2
            x = (W - largura) / 2
            for c in letras_nome:
                db.text((x, y), c, font=f, fill=(255, 255, 255, 204))
                x += db.textlength(c, font=f) + passo
            y -= 26
        else:
            fc, esp, linhas_c, medir_c = f
            y -= tam * 1.5 * len(linhas_c)
            yy = y
            for linha in linhas_c:
                _escrever_linha(db, linha, yy + tam * 0.25, fc, esp, medir_c, branco)
                yy += tam * 1.5
    texto_baixo = _sombra(texto_baixo, 14, 0.5)
    if estrelas:
        r = 13
        y -= 18 + 2 * r
        passo = 2 * r + 0.18 * 26 * 1.6
        x0 = (W - (estrelas - 1) * passo) / 2
        de = ImageDraw.Draw(texto_baixo)
        for k in range(estrelas):
            _estrela(de, x0 + k * passo, y + r, r, s.get("cor_estrelas", "#E5892B"))
    camada = Image.alpha_composite(camada, texto_baixo)
    zonas = []
    alfa = camada.getchannel("A").point(lambda v: 255 if v > 60 else 0)
    for y0, y1 in ((0, H // 2), (H // 2, H)):
        caixa = alfa.crop((0, y0, W, y1)).getbbox()
        if caixa:
            zonas.append((y0 + caixa[1] - 30, y0 + caixa[3] + 30))
    return camada, zonas


def texto_por_cima(base, s, plano=None, camada=None):
    """A foto com os véus escuros e o texto por cima ("veu_topo" 0,5 e "veu_baixo" 0,8)."""
    if camada is None:
        camada, _ = camada_de_texto(s, plano)
    im = base.convert("RGBA")
    if s.get("titulo") and s.get("veu_topo", 0.5):
        a = s.get("veu_topo", 0.5)
        im.alpha_composite(_veu(560, [(0, a), (0.5, a * 0.5), (1, 0)]), (0, 0))
    if (s.get("citacao") or s.get("assinatura")) and s.get("veu_baixo", 0.8):
        a = s.get("veu_baixo", 0.8)
        im.alpha_composite(_veu(520, [(0, a), (0.4, a * 0.69), (1, 0)], de_baixo=True), (0, H - 520))
    return Image.alpha_composite(im, camada).convert("RGB")


# ---------- layouts ----------
# Cada função devolve uma lista de slides (normalmente 1; o panorama devolve 2).

def cheia(s, p, n):
    """Foto a toda a página, recortada ao formato do slide."""
    t = tela(s, p)
    colar_preenchido(t, s["fotos"][0], (0, 0, W, H), n, s.get("_zonas", ()))
    return [t]


def inteira(s, p, n):
    """Foto inteira, sem recorte, centrada, com a margem assumida à volta.

    `margem` é a margem mínima (a foto encosta-se a ela no lado mais apertado)."""
    t = tela(s, p)
    m = s.get("margem", 120)
    im, _, _ = foto(s["fotos"][0])
    im = caber(im, W - 2 * m, H - 2 * m)
    x, y = (W - im.width) // 2, (H - im.height) // 2
    t.paste(im, (x, y))
    if s.get("pelicula") or (isinstance(s["fotos"][0], dict) and s["fotos"][0].get("pelicula")):
        pelicula(t, (x, y, im.width, im.height), semente=n)
    return [t]


def _por_cima(t, spec, caixa, n, borda=0):
    """Uma foto inteira (sem corte) a caber na caixa, centrada, com borda branca ou película."""
    x, y, w, h = caixa
    im, _, nome = foto(spec)
    im = caber(im, w - 2 * borda, h - 2 * borda)
    fx, fy = x + (w - im.width - 2 * borda) // 2, y + (h - im.height - 2 * borda) // 2
    if borda:
        t.paste("#FFFFFF", (fx, fy, fx + im.width + 2 * borda, fy + im.height + 2 * borda))
    t.paste(im, (fx + borda, fy + borda))
    if isinstance(spec, dict) and spec.get("pelicula"):
        pelicula(t, (fx + borda, fy + borda, im.width, im.height), semente=n + fx)
    return (fx, fy, im.width + 2 * borda, im.height + 2 * borda)


def _caras_tapadas(fundo, caixas, n, o_que):
    """Avisa quando uma foto posta por cima tapa uma cara da foto de fundo."""
    a = recorte.analisar(fundo)
    for cx, cy, cw, ch in a["caras"]:
        fx0, fy0, fx1, fy1 = cx * W, cy * H, (cx + cw) * W, (cy + ch) * H
        for x, y, w, h in caixas:
            if min(fx1, x + w) - max(fx0, x) > 0.3 * (fx1 - fx0) and min(fy1, y + h) - max(fy0, y) > 0.3 * (fy1 - fy0):
                AVISOS.append(f"slide {n}: {o_que} tapa uma cara da foto de fundo")
                return


def sobre_textura(s, p, n):
    """Uma foto de detalhe do casamento a toda a página (os azulejos, as flores, a toalha, a pedra)
    e, por cima, outra foto inteira. "textura": a foto de fundo; "fotos": [a foto por cima];
    "margem_x" (170) e "margem_y" (200): o espaço à volta da foto por cima; "borda": px de branco."""
    t = tela(s, p)
    colar_preenchido(t, s["textura"], (0, 0, W, H), n)
    fundo = t.copy()
    mx, my = s.get("margem_x", 170), s.get("margem_y", 200)
    caixa = _por_cima(t, s["fotos"][0], (mx, my, W - 2 * mx, H - 2 * my), n, s.get("borda", 0))
    _caras_tapadas(fundo, [caixa], n, "a foto por cima")
    return [t]


COLAGENS = {  # (x, y, largura) de cada foto, para 2 e 3 fotos desencontradas
    2: [(70, 600, 470), (570, 110, 430)],
    3: [(70, 90, 430), (590, 330, 420), (230, 830, 400)],
}


def colagem(s, p, n):
    """Duas ou três fotos inteiras, de tamanhos diferentes e desencontradas, com muito fundo à volta.
    "posicoes": [[x, y, largura], ...] para mudar o arranjo; "borda" e "pelicula" como nas outras."""
    t = tela(s, p)
    posicoes = s.get("posicoes", COLAGENS[len(s["fotos"])])
    for spec, (x, y, w) in zip(s["fotos"], posicoes):
        im, _, _ = foto(spec)
        h = round(w * im.height / im.width)
        y = min(y, H - 50 - h)
        _por_cima(t, spec, (x, max(40, y), w, h), n, s.get("borda", 0))
    return [t]


def _caras_no_slide(im, caixa):
    """As caras de uma imagem colada na caixa (x, y, w, h), em px do slide."""
    x, y, w, h = caixa
    return [(x + cx * w, y + cy * h, x + (cx + cw) * w, y + (cy + ch) * h)
            for cx, cy, cw, ch in recorte.analisar(im)["caras"]]


def _tapa(caixa, caras):
    """Quanto das caras fica por baixo da caixa (0 = nada)."""
    x, y, w, h = caixa
    total = 0.0
    for a0, b0, a1, b1 in caras:
        sx = max(0, min(a1, x + w) - max(a0, x)) / max(1, a1 - a0)
        sy = max(0, min(b1, y + h) - max(b0, y)) / max(1, b1 - b0)
        total += sx * sy
    return total


def camadas_etapas(s, p, n):
    """A colagem por camadas: uma foto a toda a página de fundo ("fundo_foto") e, por cima, as "fotos"
    a entrar uma de cada vez, com "borda" branca (14 px) ou "pelicula". Devolve uma imagem por etapa
    (só o fundo, o fundo e a 1.ª, ...); no vídeo cada etapa fica "segundos_por_camada".

    Cada foto nova procura o sítio e o tamanho onde não tapa nenhuma cara (as da foto de fundo e as
    das fotos que já lá estão), longe da foto anterior para a colagem não ficar empilhada.
    "posicoes": [[centro x, centro y, largura], ...] (de 0 a 1) para fixar os sítios à mão."""
    t = tela(s, p)
    colar_preenchido(t, s["fundo_foto"], (0, 0, W, H), n)
    caras = _caras_no_slide(t, (0, 0, W, H))
    etapas, anterior = [t.copy()], None
    for k, spec in enumerate(s["fotos"]):
        im, _, nome = foto(spec)
        if "posicoes" in s:
            candidatos = [tuple(s["posicoes"][k % len(s["posicoes"])])]
        else:
            candidatos = [(cx, cy, lw) for lw in (0.52, 0.46, 0.40) for cx in (0.3, 0.4, 0.5, 0.6, 0.7)
                          for cy in (0.32, 0.42, 0.5, 0.58, 0.68)]
        melhor = None
        for cx, cy, lw in candidatos:
            w = round(lw * W)
            h = min(round(w * im.height / im.width), round(H * 0.7))
            w = min(w, round(h * im.width / im.height))
            x = min(max(round(cx * W - w / 2), 30), W - 30 - w)
            y = min(max(round(cy * H - h / 2), 30), H - 30 - h)
            tapado = _tapa((x, y, w, h), caras)
            longe = 0 if anterior is None else min(1.0, ((x - anterior[0]) ** 2 + (y - anterior[1]) ** 2) ** 0.5 / 400)
            nota = -3 * tapado + 0.6 * lw + 0.3 * longe - 0.4 * abs(cx - 0.5)
            if melhor is None or nota > melhor[0]:
                melhor = (nota, tapado, (x, y, w, h))
        _, tapado, caixa = melhor
        if tapado > 0.3:
            AVISOS.append(f"slide {n}: a foto {nome} tapa uma cara que está por baixo")
        final = _por_cima(t, spec, caixa, n + k, s.get("borda", 14))
        anterior = final
        caras += _caras_no_slide(caber(im, final[2], final[3]), final)
        etapas.append(t.copy())
    return etapas


def camadas(s, p, n):
    return [camadas_etapas(s, p, n)[-1]]


def moldura(s, p, n):
    """Margem igual nos quatro lados, e a foto preenche a área de dentro (um passe-partout)."""
    t = tela(s, p)
    m = s.get("margem", 77)
    colar_preenchido(t, s["fotos"][0], (m, m, W - 2 * m, H - 2 * m), n)
    return [t]


def mosaico(s, p, n):
    """Composição livre de várias fotos num slide.

    "linhas": [[a], [b, c]]          cada lista é uma linha, com as fotos lado a lado
    "orientacao": "colunas"          as listas passam a ser colunas, com as fotos empilhadas
    "pesos": [2, 1]                  tamanho relativo de cada linha (ou coluna)
    "pesos_internos": [[1], [1, 2]]  tamanho relativo das fotos dentro de cada linha
    "margem": 0, "espaco": 0         margem exterior e espaço entre fotos, em px
    "margem_x": 178, "margem_y": 217 margens diferentes dos lados e em cima/baixo
                                     (para um bloco pequeno a flutuar no fundo)
    """
    t = tela(s, p)
    m, e = s.get("margem", 0), s.get("espaco", 0)
    mx, my = s.get("margem_x", m), s.get("margem_y", m)
    grupos = s["linhas"]
    colunas = s.get("orientacao") == "colunas"
    pesos = s.get("pesos", [1] * len(grupos))
    internos = s.get("pesos_internos", [[1] * len(g) for g in grupos])
    principal = W - 2 * mx if colunas else H - 2 * my
    secundario = H - 2 * my if colunas else W - 2 * mx
    for (pos, tam), grupo, pi in zip(dividir(principal, pesos, e), grupos, internos):
        for (pos2, tam2), spec in zip(dividir(secundario, pi, e), grupo):
            if colunas:
                caixa = (mx + pos, my + pos2, tam, tam2)
            else:
                caixa = (mx + pos2, my + pos, tam2, tam)
            colar_preenchido(t, spec, caixa, n)
    return [t]


def testemunho(s, p, n):
    """Uma foto a toda a página com o título, a citação do cliente, as estrelas e o nome por cima.

    Sem citação nem assinatura, é uma frase do fotógrafo sobre a foto (o slide de identificação)."""
    camada, zonas = camada_de_texto(s, p)
    t = tela(s, p)
    colar_preenchido(t, s["fotos"][0], (0, 0, W, H), n, zonas)
    return [texto_por_cima(t, s, p, camada)]


def pronto(s, p, n):
    """Um slide já feito, usado tal como está."""
    im, _, _ = foto(s["fotos"][0])
    if im.size != (W, H):
        im = preencher(im, W, H)
    return [im]


def panorama(s, p, n):
    """Uma foto horizontal espalhada por dois slides seguidos (continua ao deslizar)."""
    larga = Image.new("RGB", (2 * W, H))
    colar_preenchido(larga, s["fotos"][0], (0, 0, 2 * W, H), n)
    return [larga.crop((0, 0, W, H)), larga.crop((W, 0, 2 * W, H))]


LAYOUTS = {f.__name__: f for f in (cheia, inteira, moldura, mosaico, panorama, pronto, testemunho,
                                   sobre_textura, colagem, camadas)}
LAYOUTS["frase"] = testemunho


# ---------- principal ----------

def previsualizacao(slides, destino, legendas):
    lw = 270
    lh = round(lw * H / W)
    colunas = min(len(slides), 6)
    linhas = (len(slides) + colunas - 1) // colunas
    f = letra("texto", 15)
    folha = Image.new("RGB", (colunas * (lw + 10) + 10, linhas * (lh + 46) + 10), "#DDDDDD")
    d = ImageDraw.Draw(folha)
    for i, (s, leg) in enumerate(zip(slides, legendas)):
        x, y = 10 + (i % colunas) * (lw + 10), 10 + (i // colunas) * (lh + 46)
        folha.paste(s.resize((lw, lh), Image.LANCZOS), (x, y))
        d.text((x, y + lh + 4), leg[0], fill="black", font=letra("texto", 17))
        if leg[1]:
            d.text((x, y + lh + 24), leg[1][:44], fill="#555555", font=f)
    folha.save(destino, quality=85)


def folha_de_cortes(destino):
    """Cada foto cortada inteira, com a parte que sai escurecida e o corte a vermelho.
    É para o assistente ver o que se perdeu antes de entregar; nunca vai para o aluno."""
    if not CORTES:
        return
    colunas, cw, chh = 8, 300, 340
    linhas = (len(CORTES) + colunas - 1) // colunas
    folha = Image.new("RGB", (colunas * cw, linhas * chh), "#FFFFFF")
    d = ImageDraw.Draw(folha)
    f = letra("texto", 15)
    for i, (n, nome, mini, (x0, y0, x1, y1), avisos) in enumerate(CORTES):
        ox = (i % colunas) * cw + (cw - mini.width) // 2
        oy = (i // colunas) * chh + 4
        escuro = Image.new("RGB", mini.size, "#000000")
        mascara = Image.new("L", mini.size, 150)
        caixa = (round(x0 * mini.width), round(y0 * mini.height), round(x1 * mini.width) - 1, round(y1 * mini.height) - 1)
        ImageDraw.Draw(mascara).rectangle(caixa, fill=0)
        folha.paste(Image.composite(escuro, mini, mascara), (ox, oy))
        d.rectangle((ox + caixa[0], oy + caixa[1], ox + caixa[2], oy + caixa[3]), outline="#E0201B", width=3)
        d.text(((i % colunas) * cw + 6, oy + 302), f"{n:02d} {nome}"[:34], fill="#E0201B" if avisos else "black", font=f)
    folha.save(destino, quality=85)


def configurar(plano):
    global W, H
    formato = plano.get("formato", "4:5")
    if formato not in FORMATOS:
        sys.exit(f"Formato '{formato}' desconhecido. Opções: {', '.join(FORMATOS)}")
    W, H = FORMATOS[formato]
    LETRAS_DO_PLANO.clear()
    LETRAS_DO_PLANO.update(plano.get("letras", {}))


def renderizar(plano, so=None):
    """Gera as imagens dos slides do plano; devolve [(número, imagem, slide), ...].

    so: os números dos slides a gerar (os outros saltam-se, mas a numeração mantém-se)."""
    configurar(plano)
    out, n = [], 1
    for s in plano["slides"]:
        layout = s.get("layout")
        if layout not in LAYOUTS:
            sys.exit(f"Slide {n}: layout desconhecido '{layout}'. Opções: {', '.join(LAYOUTS)}")
        quantos = 2 if layout == "panorama" else 1
        if so is None or any(k in so for k in range(n, n + quantos)):
            for k, im in enumerate(LAYOUTS[layout](s, plano, n)):
                out.append((n + k, im, s))
        n += quantos
    return out


def nome_do_ficheiro(n, s):
    return f"{n:02d}-{s['nome']}.jpg" if s.get("nome") else f"{n:02d}.jpg"


def gravar(im, caminho):
    im.save(caminho, quality=95, subsampling=0, icc_profile=SRGB_BYTES)


def fotos_do_slide(s):
    specs = s.get("fotos", []) + [f for linha in s.get("linhas", []) for f in linha]
    return [os.path.basename(f if isinstance(f, str) else f["ficheiro"]) for f in specs]


def montar_posts(plano, pasta_banco, pasta_posts, numerados, pasta_plano):
    """Cada post é uma lista de slides do banco, pelo número ou pelo nome."""
    por_nome = {s.get("nome"): n for n, _, s in numerados if s.get("nome")}
    for post in plano.get("posts", []):
        destino = os.path.join(pasta_posts, post["nome"])
        if os.path.isdir(destino):
            shutil.rmtree(destino)
        os.makedirs(destino)
        escolhidos = []
        for ref in post["slides"]:
            n = ref if isinstance(ref, int) else por_nome.get(ref)
            achado = next(((k, im, s) for k, im, s in numerados if k == n), None)
            if not achado:
                sys.exit(f"Post {post['nome']}: não há no banco o slide {ref}.")
            escolhidos.append(achado)
        if len(escolhidos) > 10:
            print(f"Aviso: o post {post['nome']} tem {len(escolhidos)} slides; o Instagram aceita 20, "
                  "mas o método é de 6 a 10.", file=sys.stderr)
        for i, (n, im, s) in enumerate(escolhidos, 1):
            shutil.copyfile(os.path.join(pasta_banco, nome_do_ficheiro(n, s)), os.path.join(destino, f"{i:02d}.jpg"))
        previsualizacao([im for _, im, _ in escolhidos], os.path.join(pasta_plano, f"_post-{post['nome']}.jpg"),
                        [(f"{i:02d} · banco {n:02d}", s.get("nome", "")) for i, (n, _, s) in enumerate(escolhidos, 1)])
        print(f"post {post['nome']}: {len(escolhidos)} slides em {destino}")


def main():
    args = sys.argv[1:]
    if not args or len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--slides"):
        sys.exit(__doc__)
    so = {int(x) for x in args[2].split(",")} if len(args) == 3 else None
    with open(args[0], encoding="utf-8") as fh:
        plano = json.load(fh)
    # um caminho relativo conta a partir da pasta do plano
    saida = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(args[0])), os.path.expanduser(plano.get("saida", "."))))
    pasta_banco = os.path.join(saida, "banco")
    os.makedirs(pasta_banco, exist_ok=True)

    if so is None:
        # slides de uma versão anterior não podem ficar esquecidos na pasta
        for antigo in os.listdir(pasta_banco):
            if antigo[:2].isdigit() and antigo.endswith(".jpg"):
                os.remove(os.path.join(pasta_banco, antigo))
    feitos = renderizar(plano, so)
    for n, im, s in feitos:
        if so is not None:
            for antigo in os.listdir(pasta_banco):
                if antigo.startswith(f"{n:02d}") and antigo.endswith(".jpg"):
                    os.remove(os.path.join(pasta_banco, antigo))
        gravar(im, os.path.join(pasta_banco, nome_do_ficheiro(n, s)))
    print(f"{len(feitos)} slides em {pasta_banco}")

    if so is None:
        pasta_plano = os.path.dirname(os.path.abspath(args[0]))
        previsualizacao([im for _, im, _ in feitos], os.path.join(pasta_plano, "_previsualizacao.jpg"),
                        [(f"{n:02d} {s.get('nome', '')}", " · ".join(s.get("etiquetas", []))) for n, _, s in feitos])
        folha_de_cortes(os.path.join(pasta_plano, "_cortes.jpg"))
        with open(os.path.join(pasta_plano, "_banco.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["numero", "ficheiro", "layout", "etiquetas", "fotos"])
            for n, _, s in feitos:
                w.writerow([n, nome_do_ficheiro(n, s), s["layout"], " ".join(s.get("etiquetas", [])),
                            " ".join(fotos_do_slide(s))])
        montar_posts(plano, pasta_banco, os.path.join(saida, "posts"), feitos, pasta_plano)
    else:
        print("Só refiz estes slides. Corre sem --slides para atualizar a pré-visualização e os posts.")

    if AVISOS:
        print("\nRecortes a confirmar (pés, cabeças, mãos):")
        for a in AVISOS:
            print("  " + a)


if __name__ == "__main__":
    main()

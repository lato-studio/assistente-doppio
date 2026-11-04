"""Escolhe onde cortar uma foto para caber numa caixa sem estragar o que importa.

O que importa, por ordem:
1. as caras (com o cabelo por cima) nunca ficam cortadas a meio, nem por baixo do texto;
2. as pessoas inteiras: a cabeça fica sempre, e os pés ou ficam inteiros ou o corte passa
   bem acima deles (pelas coxas ou pelos joelhos), nunca à justa;
3. o resto da história: o corte guarda a parte da foto com mais detalhe (mãos, gestos, o vestido,
   o cenário) e deita fora o que é vazio (parede, céu liso, chão).

Usa o OpenCV para encontrar caras e pessoas. Sem o OpenCV, usa só o detalhe.
O resultado é uma proposta: o assistente confere sempre os slides com os olhos.
"""
import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None

LADO = 640  # a deteção corre numa cópia pequena
_CACHE = {}


def _detetores():
    if cv2 is None:
        return None
    if "det" not in _CACHE:
        base = cv2.data.haarcascades
        _CACHE["det"] = {
            "frente": cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml"),
            "frente2": cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml"),
            "perfil": cv2.CascadeClassifier(base + "haarcascade_profileface.xml"),
            "tronco": cv2.CascadeClassifier(base + "haarcascade_upperbody.xml"),
        }
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _CACHE["det"]["hog"] = hog
    return _CACHE["det"]


def _juntar(caixas):
    """Junta as caixas que se sobrepõem muito (a mesma cara encontrada por dois detetores)."""
    out = []
    for c in sorted(caixas, key=lambda c: -c[2] * c[3]):
        x, y, w, h = c
        if not any(abs(x + w / 2 - (a + c2 / 2)) < max(w, c2) / 2 and abs(y + h / 2 - (b + d / 2)) < max(h, d) / 2
                   for a, b, c2, d in out):
            out.append(c)
    return out


def analisar(im):
    """Caras, pessoas e mapa de detalhe, em coordenadas de 0 a 1 da foto."""
    chave = (im.size, im.tobytes()[:4096:7])
    if chave in _CACHE:
        return _CACHE[chave]
    escala = LADO / max(im.size)
    pequena = im.convert("L").resize((max(1, round(im.width * escala)), max(1, round(im.height * escala))))
    pw, ph = pequena.size
    cinza = np.asarray(pequena)

    # o detalhe: a força das arestas, esbatida, para valer zonas e não pixels soltos
    a = cinza.astype(np.float32)
    gx = np.abs(np.diff(a, axis=1, append=a[:, -1:]))
    gy = np.abs(np.diff(a, axis=0, append=a[-1:, :]))
    detalhe = Image.fromarray(np.clip(gx + gy, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(pw / 40))
    detalhe = np.asarray(detalhe).astype(np.float32) + 1.0

    caras, pessoas = [], []
    det = _detetores()
    if det is not None:
        minimo = max(14, round(min(pw, ph) * 0.03))
        achadas = []
        for nome in ("frente", "frente2", "perfil"):
            achadas += [tuple(c) for c in det[nome].detectMultiScale(cinza, 1.08, 6, minSize=(minimo, minimo))]
        espelho = np.ascontiguousarray(cinza[:, ::-1])
        for x, y, w, h in det["perfil"].detectMultiScale(espelho, 1.08, 6, minSize=(minimo, minimo)):
            achadas.append((pw - x - w, y, w, h))
        for x, y, w, h in _juntar(achadas):
            # a cara com o cabelo por cima e o queixo por baixo
            caras.append(((x - 0.15 * w) / pw, (y - 0.45 * h) / ph, 1.3 * w / pw, 1.7 * h / ph))
        # as pessoas: o detetor precisa de uma cópia maior, para encontrar quem está pequeno na foto
        e2 = 1100 / max(im.size)
        grande = np.asarray(im.convert("L").resize((round(im.width * e2), round(im.height * e2))))
        gw, gh = grande.shape[1], grande.shape[0]
        rects, pesos = det["hog"].detectMultiScale(grande, winStride=(8, 8), padding=(16, 16), scale=1.05)
        for (x, y, w, h), p in zip(rects, np.ravel(pesos) if len(rects) else []):
            if p > 0.5:
                pessoas.append((x / gw, y / gh, w / gw, h / gh))
        # o tronco com a cabeça, para quem tem a cara tapada pelas mãos ou de costas
        for x, y, w, h in det["tronco"].detectMultiScale(cinza, 1.05, 4, minSize=(minimo * 2, minimo * 2)):
            caras.append((x / pw, y / ph, w / pw, 0.45 * h / ph))
    r = {"caras": caras, "pessoas": pessoas, "detalhe": detalhe}
    _CACHE[chave] = r
    return r


def _dentro(a0, a1, c0, c1):
    """Que parte do intervalo [a0, a1] (a cara) fica dentro do corte [c0, c1]."""
    return max(0.0, min(a1, c1) - max(a0, c0)) / max(1e-6, a1 - a0)


def escolher(im, largura, altura, foco=None, zonas=()):
    """A caixa (x0, y0, x1, y1), em px da foto, a recortar para caber em largura x altura.

    foco: (x, y) de 0 a 1 que o assistente pediu; o corte fica o mais perto possível dele,
    sem estragar caras nem pés. zonas: faixas (y0, y1) de 0 a 1 da caixa onde vai haver texto.
    Devolve também os avisos do que não deu para salvar."""
    W0, H0 = im.size
    escala = max(largura / W0, altura / H0)
    cw, ch = largura / escala, altura / escala
    vertical = H0 - ch > W0 - cw  # o corte é em cima e em baixo
    folga = (H0 - ch) if vertical else (W0 - cw)
    tam = (ch / H0) if vertical else (cw / W0)  # o tamanho do corte, de 0 a 1
    if folga < 1:
        return (0, 0, W0, H0), []

    a = analisar(im)
    perfil = a["detalhe"].sum(axis=1 if vertical else 0)
    acumulado = np.concatenate([[0], np.cumsum(perfil)])
    n = len(perfil)
    total = acumulado[-1]

    melhor, aviso_melhor = None, []
    for t in np.linspace(0, 1, 81):
        c0 = t * (1 - tam)
        c1 = c0 + tam
        i0, i1 = round(c0 * n), round(c1 * n)
        nota = (acumulado[i1] - acumulado[i0]) / total
        avisos = []
        for cx, cy, cwf, chf in a["caras"]:
            a0, a1 = (cy, cy + chf) if vertical else (cx, cx + cwf)
            f = _dentro(a0, a1, c0, c1)
            peso = 1.0 if (chf if vertical else cwf) > 0.04 else 0.4
            if 0.02 < f < 0.97:
                nota -= 4 * peso
                avisos.append("uma cara fica cortada")
            elif f >= 0.97 and vertical and cy - c0 < 0.02:
                nota -= 1.5 * peso  # a cabeça encostada ao topo, sem ar por cima
                avisos.append("uma cabeça fica encostada ao topo")
            elif f <= 0.02:
                nota -= 1.2 * peso
                avisos.append("uma cara fica de fora")
            if vertical and f > 0.5 and zonas:
                # a cara, em coordenadas da caixa, não pode cair numa zona de texto
                y0, y1 = (cy - c0) / tam, (cy + chf - c0) / tam
                if any(min(y1, z1) - max(y0, z0) > 0 for z0, z1 in zonas):
                    nota -= 4 * peso
                    avisos.append("o texto tapa uma cara")
        for px, py, pwf, phf in a["pessoas"]:
            if not vertical:
                if 0.05 < _dentro(px, px + pwf, c0, c1) < 0.6:
                    nota -= 0.8
                    avisos.append("uma pessoa fica cortada ao meio na largura")
                continue
            cabeca = (py, py + 0.18 * phf)
            if _dentro(*cabeca, c0, c1) < 0.95 and _dentro(py, py + phf, c0, c1) > 0.3:
                nota -= 3
                avisos.append("uma cabeça fica cortada")
            pes = py + phf
            if pes - 0.13 * phf < c1 < pes - 0.01 * phf:
                nota -= 2
                avisos.append("os pés ficam cortados à justa")
        if foco is not None:
            alvo = (foco[1] if vertical else foco[0])
            nota -= 4 * abs((c0 + c1) / 2 - alvo)  # o foco foi pedido depois de ver a foto: pesa mais
        if melhor is None or nota > melhor[0]:
            melhor, aviso_melhor = (nota, c0), sorted(set(avisos))
    c0 = melhor[1]
    if vertical:
        y0 = round(c0 * H0)
        return (0, y0, W0, min(H0, y0 + round(ch))), aviso_melhor
    x0 = round(c0 * W0)
    return (x0, 0, min(W0, x0 + round(cw)), H0), aviso_melhor


def recortar(im, largura, altura, foco=None, zonas=()):
    caixa, avisos = escolher(im, largura, altura, foco, zonas)
    return im.crop(caixa).resize((largura, altura), Image.LANCZOS), caixa, avisos

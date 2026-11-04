"""Monta um Reel 9:16 a partir de um plano JSON (o motor da skill de Reels do Assistente Doppio).

Os cortes caem nas batidas da música. Um plano pode durar batidas inteiras ou
frações (0.5, 0.25) para o modo rápido. As fotos nunca mudam de cor nem se
mexem: só se recortam para 9:16, à volta do ponto de interesse, ou entram numa
moldura quando são horizontais. Sem música, usa um ritmo fixo e exporta sem som,
para o aluno escolher o áudio dentro do Instagram.

Uso: python reel.py <plano.json>

Plano:
{
  "musica": "faixa.mp3" | null,       sem música: ritmo fixo ("bpm", 120 por defeito)
  "inicio_musica": 12.5,              opcional: começa no trecho mais perto deste segundo
  "revelar_no_drop": 1,               opcional: o plano n.º 1 (a contar do 0) entra no instante exato do drop da música
  "musica_baixa_antes_do_drop": 0.3,  com revelar_no_drop: o volume da música até ao drop (1 = sem baixar)
  "volume_teclas": 0.35, "volume_disparo": 1.4,
  "som_disparo": "disparo.wav",       opcional: o som da câmara do aluno, em vez do do Sérgio
  "voz": {"ficheiro": "audio.mp3", "palavras": "transcricao.json", "trechos": [[1.7, 4.5], [16.3, 21.3]],
          "inicio": 1.2, "pausa": 0.4, "musica_por_baixo": 0.2, "musica_nas_pausas": 0.85}
                                      um trecho [de, até, 4] deixa 4 batidas só de música antes do seguinte
                                      Reels de prova: a voz do cliente, com legendas palavra a palavra
                                      (a transcrição é a do whisper.cpp com -ojf -dtw)
  "pasta_fotos": "...", "pasta_bastidores": "...",
  "logo": "logo.png", "largura_logo": 380, "fundo_logo": [0, 0, 0], "convite": "Datas para 2027 · link na bio",
  "letra": "caminho/para/letra.ttf",
  "saida": "Reel.mp4",                opcional: por defeito vai para o Ambiente de Trabalho
  "planos": [
    {"foto": "a.jpg", "batidas": 3, "foco": [0.5, 0.4]},
    {"foto": "b.jpg", "batidas": 1, "enquadramento": "moldura"},
    {"fotos": ["c.jpg", "d.jpg"], "batidas": 1},                 duas horizontais empilhadas
    {"bastidor": "IMG_1.MOV", "inicio": 1.0, "batidas": 4},
    {"foto": "e.jpg", "batidas": 2, "texto": ["Ninguém parou", "para estas fotos."]},
    {"bastidor": "IMG_2.MOV", "batidas": 6, "texto": ["POV:", "Quando..."], "escrever": true, "velocidade": 24}
                                      escrever: letra a letra, com o clique de uma tecla em cada uma; o mesmo
                                      texto em planos seguidos continua de onde ia
  Cada foto que aparece logo depois de um bastidor leva o som de um disparo de câmara.
  ]
}
"""
import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import pillow_heif
from PIL import Image, ImageCms, ImageDraw, ImageFilter, ImageFont, ImageOps

pillow_heif.register_heif_opener()
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1080, 1920, 30
LOGO_S, FADE_S = 1.6, 0.6


# ---------- música ----------

def batidas_da_musica(faixa):
    y, sr = librosa.load(faixa, sr=22050)
    envelope = librosa.onset.onset_strength(y=y, sr=sr)
    _, batidas = librosa.beat.beat_track(onset_envelope=envelope, sr=sr, units="time", tightness=200)
    ataques = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, units="time")
    rms = librosa.feature.rms(y=y)[0]
    t_rms = librosa.times_like(rms, sr=sr)
    # a energia de cada batida é a média do intervalo até à seguinte (mais estável do que o valor num instante)
    limites = np.append(batidas, len(y) / sr)
    energia = np.array([rms[(t_rms >= a) & (t_rms < b)].mean() if np.any((t_rms >= a) & (t_rms < b)) else 0.0
                        for a, b in zip(limites[:-1], limites[1:])])
    return np.asarray(batidas), ataques, len(y) / sr, energia


def batida_do_drop(energia, minimo=8, folga=40):
    """A batida onde a música explode: a maior subida das duas batidas seguintes face às duas anteriores."""
    fim = max(minimo + 1, len(energia) - folga)
    i = np.arange(minimo, fim)
    depois = (energia[i] + energia[np.minimum(i + 1, len(energia) - 1)]) / 2
    antes = (energia[i - 1] + energia[i - 2]) / 2
    return minimo + int(np.argmax(depois / (antes + 1e-4)))


def tempo_na_grelha(batidas, posicao):
    """Tempo (s) de uma posição fracionária na grelha de batidas (ex.: 12.5 = a meio entre a 12 e a 13)."""
    i = int(np.floor(posicao))
    if i + 1 >= len(batidas):
        raise SystemExit("O plano é mais longo do que a música. Encurta o plano ou escolhe um início mais cedo.")
    return batidas[i] + (posicao - i) * (batidas[i + 1] - batidas[i])


# ---------- material ----------

def abrir_foto(caminho):
    """Lê JPG/PNG/HEIC e converte para sRGB quando o perfil é outro (ex.: Display P3 do iPhone). A cor não muda."""
    im = ImageOps.exif_transpose(Image.open(caminho))
    icc = im.info.get("icc_profile")
    if icc:
        perfil = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if "sRGB" not in ImageCms.getProfileDescription(perfil):
            im = ImageCms.profileToProfile(im, perfil, ImageCms.createProfile("sRGB"), renderingIntent=0, outputMode="RGB")
    return np.array(im.convert("RGB"))


def cheio(img, foco):
    """Recorte 9:16 a encher o ecrã, com o ponto de interesse o mais ao centro possível."""
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    cx = min(max(foco[0] * w, W / (2 * s)), w - W / (2 * s))
    cy = min(max(foco[1] * h, H / (2 * s)), h - H / (2 * s))
    M = np.array([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)


def moldura(img, fundo=(0, 0, 0)):
    """Foto inteira, sem corte, centrada sobre o fundo (para horizontais que não aguentam o recorte)."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nova = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    tela = np.full((H, W, 3), fundo, np.uint8)
    y, x = (H - nova.shape[0]) // 2, (W - nova.shape[1]) // 2
    tela[y:y + nova.shape[0], x:x + nova.shape[1]] = nova
    return tela


def dupla(a, b, junta=8):
    """Duas horizontais empilhadas, cada uma com metade do ecrã."""
    meia = (H - junta) // 2
    partes = []
    for img in (a, b):
        h, w = img.shape[:2]
        s = max(W / w, meia / h)
        r = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        y, x = (r.shape[0] - meia) // 2, (r.shape[1] - W) // 2
        partes.append(r[y:y + meia, x:x + W])
    tela = np.zeros((H, W, 3), np.uint8)
    tela[:meia] = partes[0]
    tela[meia + junta:meia + junta + meia] = partes[1]
    return tela


def filtro_sdr(caminho):
    """Os vídeos do iPhone vêm em HDR (HLG). Sem conversão ficam lavados; com ela ficam como no telemóvel."""
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", str(caminho)], capture_output=True, text=True).stderr
    if "arib-std-b67" in info or "smpte2084" in info:
        return ("zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
                "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,")
    return ""


def frames_do_bastidor(caminho, inicio, n):
    cmd = [FFMPEG, "-v", "error", "-ss", f"{inicio:.3f}", "-i", str(caminho), "-frames:v", str(n),
           "-vf", f"{filtro_sdr(caminho)}fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}",
           "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    if len(frames) == 0:
        raise SystemExit(f"Não consegui ler o bastidor {caminho.name} a partir de {inicio}s.")
    if len(frames) < n:
        frames = np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), 0)])
    return frames


# ---------- texto ----------

def letra(caminho, tamanho, indice=0):
    # sem letra na ficha, a Avenir Next Demi Bold no Mac (índice 2 do .ttc) e a Segoe UI Semibold no Windows
    candidatos = [(caminho, indice)] if caminho else []
    candidatos += [("/System/Library/Fonts/Avenir Next.ttc", 2), ("C:/Windows/Fonts/seguisb.ttf", 0),
                   ("C:/Windows/Fonts/segoeui.ttf", 0), ("DejaVuSans.ttf", 0)]
    for c, i in candidatos:
        try:
            return ImageFont.truetype(str(Path(c).expanduser()), tamanho, index=i)
        except OSError:
            continue
    return ImageFont.load_default(tamanho)


def camada_de_texto(linhas, fnt, cor=(255, 255, 255), y_centro=0.24, sombra=0.7, entrelinha=1.25, letras=None):
    """letras: quantas letras já estão escritas (para o efeito de máquina de escrever); None escreve tudo."""
    camada = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(camada)
    y = H * y_centro - fnt.size * entrelinha * len(linhas) / 2
    falta = sum(len(l) for l in linhas) if letras is None else letras
    for linha in linhas:
        if falta > 0:
            d.text(((W - d.textlength(linha, font=fnt)) / 2, y), linha[:falta], font=fnt, fill=cor)
        falta -= len(linha)
        y += fnt.size * entrelinha
    if sombra:
        a = Image.fromarray(np.array(camada)[..., 3]).filter(ImageFilter.GaussianBlur(fnt.size * 0.35))
        base = np.zeros((H, W, 4), np.uint8)
        base[..., 3] = (np.array(a).astype(np.float32) * sombra).astype(np.uint8)
        camada = Image.alpha_composite(Image.fromarray(base), camada)
    return np.array(camada)


def por_cima(frame, camada, alfa=1.0):
    a = camada[..., 3:4].astype(np.float32) / 255.0 * alfa
    return (frame.astype(np.float32) * (1 - a) + camada[..., :3].astype(np.float32) * a).astype(np.uint8)


# ---------- som ----------

SR = 48000


def clique_de_maquina(rng, sr=SR):
    """Uma tecla de máquina de escrever: um estalido seco e curto, um pouco diferente de cada vez. Gerado aqui, sem direitos."""
    n = int(0.018 * sr)
    ruido = np.diff(rng.standard_normal(n + 1)) * np.exp(-np.linspace(0, 9, n))
    tique = np.sin(2 * np.pi * rng.uniform(1700, 2300) * np.arange(n) / sr) * np.exp(-np.linspace(0, 14, n)) * 0.6
    som = ruido + tique
    return som / np.max(np.abs(som)) * rng.uniform(0.75, 1.0)


def disparo_amostras(sr=SR):
    """O som de uma câmara reflex a disparar: "tchac-tchac", o espelho a subir com a cortina a abrir, e depois
    o espelho a descer. Cada pancada tem um corpo grave e ressonâncias metálicas de 1 a 5 kHz, com cerca de 40 ms,
    para não soar a clique de rato (pedido do Sérgio). Gerado aqui, sem direitos."""
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(11)
    som = np.zeros(int(0.22 * sr))
    passa_banda = butter(2, [700, 6500], btype="band", fs=sr, output="sos")

    def pancada(forca, ressonancias, grave, dur=0.045):
        n = int(dur * sr)
        t = np.arange(n) / sr
        ataque = sosfilt(passa_banda, rng.standard_normal(n)) * np.exp(-t / 0.006)
        metal = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3)) * np.exp(-t / d) for f, a, d in ressonancias)
        corpo = np.sin(2 * np.pi * grave * t) * np.exp(-t / 0.018) * 0.9
        cauda = sosfilt(passa_banda, rng.standard_normal(n)) * np.exp(-t / 0.02) * 0.25
        return (ataque * 1.2 + metal + corpo + cauda) * forca

    abrir = pancada(1.0, [(1350, 0.5, 0.012), (2400, 0.45, 0.009), (3900, 0.3, 0.007), (5200, 0.15, 0.005)], 110)
    fechar = pancada(0.8, [(1150, 0.45, 0.014), (2150, 0.4, 0.010), (3600, 0.25, 0.008)], 95, dur=0.06)
    for inicio, parte in ((0.0, abrir), (0.085, fechar)):
        i = int(inicio * sr)
        som[i:i + len(parte)] += parte[:len(som) - i]
    return som / np.max(np.abs(som)) * 0.9


def gravar_wav(destino, som, sr=SR):
    """Mono (n,) ou estéreo (n, 2)."""
    som = np.asarray(som)
    canais = 1 if som.ndim == 1 else som.shape[1]
    pcm = (np.clip(som, -1, 1) * 32767).astype("<i2")
    cabecalho = b"RIFF" + (36 + pcm.nbytes).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little") + \
        (1).to_bytes(2, "little") + canais.to_bytes(2, "little") + sr.to_bytes(4, "little") + \
        (sr * 2 * canais).to_bytes(4, "little") + (2 * canais).to_bytes(2, "little") + (16).to_bytes(2, "little") + \
        b"data" + pcm.nbytes.to_bytes(4, "little")
    Path(destino).write_bytes(cabecalho + pcm.tobytes())


# ---------- voz do cliente (Reels de prova) ----------

def ler_audio(caminho, sr=SR, canais=1, inicio=0.0, duracao=None):
    """Qualquer áudio ou vídeo, pelo ffmpeg (há ficheiros .mp3 que são AAC por dentro)."""
    corte = ["-ss", f"{inicio:.3f}"] + (["-t", f"{duracao:.3f}"] if duracao else [])
    pcm = subprocess.run([FFMPEG, "-v", "error", *corte, "-i", str(Path(caminho).expanduser()), "-ac", str(canais),
                          "-ar", str(sr), "-f", "s16le", "-"], capture_output=True, check=True).stdout
    som = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
    return som if canais == 1 else som.reshape(-1, canais)


def palavras_do_whisper(caminho):
    """As palavras de uma transcrição do whisper.cpp (-ojf -dtw), com o segundo em que cada uma começa."""
    dados = json.loads(Path(caminho).expanduser().read_text(encoding="utf-8"))
    palavras = []
    for seg in dados["transcription"]:
        for t in seg["tokens"]:
            texto = t["text"]
            if texto.startswith("[") or not texto.strip():
                continue
            inicio = (t["t_dtw"] / 100 - 0.12) if t.get("t_dtw", -1) >= 0 else t["offsets"]["from"] / 1000
            if texto.startswith(" ") or not palavras:
                palavras.append([texto.strip(), inicio])
            else:
                palavras[-1][0] += texto  # o resto de uma palavra partida em tokens, ou a pontuação
    return palavras


def montar_voz(voz, batidas_reel=None):
    """Junta os trechos escolhidos da voz. Entre dois trechos fica uma pausa curta, ou, se o trecho disser
    [de, até, n], n batidas só de música: a voz cala-se, a música sobe, e a voz volta a entrar numa batida.
    Devolve as amostras, as palavras de cada trecho no tempo do Reel e o segundo em que a voz acaba."""
    fonte = ler_audio(voz["ficheiro"])
    palavras = palavras_do_whisper(voz["palavras"]) if voz.get("palavras") else []
    t, pausa, partes, ditas = voz.get("inicio", 1.0), voz.get("pausa", 0.4), [], []
    for trecho in voz["trechos"]:
        a, b = trecho[:2]
        parte = fonte[int(a * SR):int(b * SR)].copy()
        f = int(0.03 * SR)
        parte[:f] *= np.linspace(0, 1, f)
        parte[-f:] *= np.linspace(1, 0, f)
        partes.append((t, parte))
        ditas.append([[w, t + (ini - a)] for w, ini in palavras if a - 0.1 <= ini < b])
        fim = t + (b - a)
        if len(trecho) > 2 and trecho[2] and batidas_reel is not None:
            seguinte = int(np.searchsorted(batidas_reel, fim - 0.05)) + int(trecho[2])
            t = float(batidas_reel[min(seguinte, len(batidas_reel) - 1)])
        else:
            t = fim + pausa
    return partes, ditas, fim


LIGACOES = {"porque", "e", "mas", "depois", "então", "quando"}


def frases_das_legendas(ditas, max_letras=48):
    """Parte cada trecho nas frases ditas. Uma frase longa parte-se numa vírgula ou antes de um "porque", "e",
    "mas"... se ficar perto do meio; senão, na palavra do meio. Nunca fica uma palavra sozinha no ecrã."""
    def comprimento(fr):
        return len(" ".join(w for w, _ in fr))

    def partir(frase):
        if comprimento(frase) <= max_letras or len(frase) < 4:
            return [frase]
        meio = comprimento(frase) / 2
        pos = lambda k: comprimento(frase[:k])
        bons = [k for k in range(2, len(frase) - 1)
                if (frase[k - 1][0].endswith(",") or frase[k][0].lower().strip(",.") in LIGACOES)
                and abs(pos(k) - meio) <= 0.2 * comprimento(frase)]
        todos = range(2, len(frase) - 1)
        k = min(bons or todos, key=lambda k: abs(pos(k) - meio))
        return partir(frase[:k]) + partir(frase[k:])

    frases = []
    for trecho in ditas:
        if trecho:
            trecho[-1][0] = trecho[-1][0].rstrip(",;")  # o trecho foi cortado aqui: não acaba numa vírgula
        atual = []
        for w, t in trecho:
            atual.append([w, t])
            if w[-1] in ".?!":
                frases += partir(atual)
                atual = []
        if atual:
            frases += partir(atual)
    return frases


def linhas_da_frase(palavras, max_letras=26):
    """Uma linha se couber; senão, duas linhas o mais iguais possível."""
    texto = " ".join(palavras)
    if len(texto) <= max_letras or len(palavras) < 2:
        return [texto]
    k = min(range(1, len(palavras)), key=lambda k: max(len(" ".join(palavras[:k])), len(" ".join(palavras[k:]))))
    return [" ".join(palavras[:k]), " ".join(palavras[k:])]


# ---------- montagem ----------

def montar(caminho_plano):
    plano = json.loads(Path(caminho_plano).read_text(encoding="utf-8"))
    planos = plano["planos"]
    total_batidas = sum(p["batidas"] for p in planos)
    pasta_fotos = Path(plano["pasta_fotos"]).expanduser()
    pasta_bast = Path(plano.get("pasta_bastidores") or ".").expanduser()

    faixa = plano.get("musica")
    if faixa:
        faixa = Path(faixa).expanduser()
        batidas, ataques, dur_musica, energia = batidas_da_musica(faixa)
    else:
        bpm = plano.get("bpm", 120)
        batidas = np.arange(0, 600, 60 / bpm)
        ataques, dur_musica = batidas, batidas[-1]
    batida = float(np.median(np.diff(batidas)))
    revelar = plano.get("revelar_no_drop") if faixa else None
    posicoes = np.cumsum([0] + [p["batidas"] for p in planos])  # em batidas, desde o início do Reel

    if revelar is not None:
        # a foto revelada entra no instante exato em que a música explode (pedido do Sérgio):
        # acha-se a batida do drop e acerta-se ao ataque mais próximo, ao centésimo de segundo
        antes = posicoes[revelar]
        drop = batida_do_drop(energia, minimo=max(8, int(np.ceil(antes)) + 2))
        perto = ataques[np.abs(ataques - batidas[drop]) < 0.12]
        t_drop = float(perto[np.argmin(np.abs(perto - batidas[drop]))]) if len(perto) else float(batidas[drop])
        desvio = t_drop - batidas[drop]
        t0 = t_drop - antes * batida
        if t0 < 0:
            raise SystemExit(f"Os bastidores antes do drop são mais longos do que a música até ao drop ({t_drop:.1f}s). Encurta-os.")
        # antes do drop os bastidores medem-se em batidas médias; depois, segue-se a grelha da música
        tempos = [t0 + b * batida if b <= antes else tempo_na_grelha(batidas, drop + b - antes) + desvio for b in posicoes]
        print(f"drop na batida {drop}, aos {t_drop:.2f}s da música (a foto {revelar} entra aí)")
    else:
        i0 = 0 if not faixa else int(np.argmin(np.abs(batidas - plano.get("inicio_musica", batidas[min(8, len(batidas) - 1)]))))
        tempos = [tempo_na_grelha(batidas, i0 + b) for b in posicoes]
        t0 = batidas[i0]
    cortes = [round((t - t0) * FPS) for t in tempos]
    t_logo = tempos[-1] - t0
    total = min(t_logo + LOGO_S, dur_musica - t0)
    n_total = round(total * FPS)
    print(f"batida {batida:.3f}s ({60 / batida:.0f} bpm), início {t0:.2f}s, {len(planos)} planos, "
          f"{total_batidas:g} batidas, logo aos {t_logo:.2f}s, duração {total:.1f}s")

    # Reels de prova: a voz do cliente por cima, e as legendas a aparecer palavra a palavra
    voz_partes, legendas, fim_voz = [], [], None
    if plano.get("voz"):
        voz_partes, ditas, fim_voz = montar_voz(plano["voz"], np.asarray(batidas) - t0)
        print("voz:", ", ".join(f"{t:.2f}-{t + len(v) / SR:.2f}s" for t, v in voz_partes))
        if fim_voz > t_logo:
            print(f"aviso: a voz acaba aos {fim_voz:.1f}s, depois do logo ({t_logo:.1f}s). Acrescenta batidas ao plano.")
        fnt_leg = letra(plano.get("letra_legendas"), plano.get("tamanho_legendas", 64), plano.get("letra_legendas_indice", 0))
        frases = frases_das_legendas(ditas)
        for i, fr in enumerate(frases):
            seguinte = frases[i + 1][0][1] if i + 1 < len(frases) else fim_voz + 0.3
            # a legenda sai quando a voz se cala: nas pausas de música fica só a foto
            fim_trecho = next((t + len(v) / SR for t, v in voz_partes if t - 0.2 <= fr[0][1] < t + len(v) / SR), fim_voz)
            ate = min(seguinte, fr[-1][1] + 1.6, fim_trecho + 0.3)
            legendas.append({"palavras": fr, "ate": ate, "linhas": linhas_da_frase([w for w, _ in fr]), "camadas": {}})

    def com_legenda(fr, g):
        t = g / FPS
        for lg in legendas:
            if lg["palavras"][0][1] <= t < lg["ate"]:
                n = sum(1 for _, tw in lg["palavras"] if tw <= t)
                letras = sum(len(w) + 1 for w, _ in lg["palavras"][:n])
                # as linhas ficam no lugar final desde a primeira palavra; contam-se as letras de cada palavra já dita
                ate_aqui, cont = [], 0
                for linha in lg["linhas"]:
                    ate_aqui.append(linha[:max(0, min(len(linha), letras - cont))])
                    cont += len(linha) + 1
                if n not in lg["camadas"]:
                    lg["camadas"][n] = camada_de_texto(lg["linhas"], fnt_leg, y_centro=plano.get("legendas_y", 0.66),
                                                       sombra=0.85, letras=sum(len(l) for l in ate_aqui))
                return por_cima(fr, lg["camadas"][n])
        return fr

    fundo_logo = tuple(plano.get("fundo_logo", [0, 0, 0]))
    # o Reel acabado vai sempre para o Ambiente de Trabalho do aluno (pedido do Sérgio)
    ambiente = next((d for d in (Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop",
                                 Path.home() / "Ambiente de Trabalho") if d.exists()), Path.home())
    saida = ambiente / Path(plano.get("saida", "Reel.mp4")).name
    saida.parent.mkdir(parents=True, exist_ok=True)
    so_video = saida.with_suffix(".video.mp4")
    enc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p,"
                "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
         "-movflags", "+faststart", str(so_video)], stdin=subprocess.PIPE)

    def escrever(fr):
        enc.stdin.write(np.ascontiguousarray(fr).tobytes())

    # o logo: pequeno, com uma sombra suave para se ler por cima da foto
    logo = Image.open(Path(plano["logo"]).expanduser()).convert("RGBA")
    lw = plano.get("largura_logo", 380)
    logo = logo.resize((lw, round(logo.height * lw / logo.width)), Image.LANCZOS)
    camada_logo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    camada_logo.paste(logo, ((W - lw) // 2, (H - logo.height) // 2), logo)
    if plano.get("convite"):
        convite = camada_de_texto([plano["convite"]], letra(plano.get("letra"), 30),
                                  y_centro=0.5 + logo.height / H / 2 + 0.04, sombra=0)
        camada_logo = Image.alpha_composite(camada_logo, Image.fromarray(convite))
    alfa = Image.fromarray(np.array(camada_logo)[..., 3]).filter(ImageFilter.GaussianBlur(18))
    sombra = np.zeros((H, W, 4), np.uint8)
    sombra[..., 3] = (np.array(alfa).astype(np.float32) * 0.8).astype(np.uint8)
    camada_logo = np.array(Image.alpha_composite(Image.fromarray(sombra), camada_logo))

    # o logo entra com um flash e um disparo de câmara, umas batidas antes do fim da última foto
    # (pedido do Sérgio: o fecho tem de ter energia, sem ecrã preto à espera)
    antes = round(plano.get("logo_antes_batidas", 2) * batida * FPS)
    f_logo = max(cortes[-2] + 3, cortes[-1] - antes)
    FLASH = [0.75, 0.4, 0.15, 0.05]

    escuro = plano.get("escuro_fecho", 0.7)

    def com_logo(fr, g):
        k = g - f_logo
        if k < 0:
            return fr
        # a foto começa a escurecer no mesmo instante em que o logo entra
        a = escuro * min(1.0, k / (0.5 * FPS))
        fr = (fr.astype(np.float32) * (1 - a) + np.array(fundo_logo, np.float32) * a).astype(np.uint8)
        fr = por_cima(fr, camada_logo, min(1.0, (k + 1) / 3))
        if k < len(FLASH):
            fr = (fr.astype(np.float32) + (255 - fr.astype(np.float32)) * FLASH[k]).astype(np.uint8)
        return fr

    ultimo, escritos = None, 0
    sons, escrita = [(f_logo / FPS, "disparo")], {}  # (segundo, tipo) dos efeitos gerados pelo motor
    for k, p in enumerate(planos):
        n = cortes[k + 1] - cortes[k]
        if n <= 0:
            continue
        if "bastidor" in p:
            frames = frames_do_bastidor(pasta_bast / p["bastidor"], p.get("inicio", 0.0), n)
        else:
            if "fotos" in p:
                img = dupla(*(abrir_foto(pasta_fotos / f) for f in p["fotos"][:2]))
            else:
                foto = abrir_foto(pasta_fotos / p["foto"])
                img = moldura(foto, fundo_logo) if p.get("enquadramento") == "moldura" else cheio(foto, p.get("foco", [0.5, 0.42]))
            frames = [img] * n
        # o disparo de câmara marca cada foto que aparece depois de um bastidor (o "resultado")
        if k > 0 and "bastidor" in planos[k - 1] and "bastidor" not in p:
            sons.append((cortes[k] / FPS, "disparo"))
        texto, fnt = None, None
        if p.get("texto"):
            fnt = letra(plano.get("letra"), p.get("tamanho", 62), plano.get("letra_indice", 0))
            chave = (tuple(p["texto"]), p.get("texto_y", 0.24))
            if p.get("escrever"):
                # máquina de escrever: o mesmo texto em planos seguidos continua a escrever-se de onde ia
                if escrita.get("chave") != chave:
                    escrita.update(chave=chave, inicio=escritos, letras=0, camadas={})
            else:
                texto = camada_de_texto(p["texto"], fnt, y_centro=p.get("texto_y", 0.24), sombra=p.get("sombra", 0.7))
        if not p.get("escrever"):
            escrita.clear()
        for j, fr in enumerate(frames):
            if p.get("escrever") and p.get("texto"):
                todas = sum(len(l) for l in p["texto"])
                n_letras = min(todas, int((escritos - escrita["inicio"]) / FPS * p.get("velocidade", 24)) + 1)
                plano_letras = "".join(p["texto"])
                for c in range(escrita["letras"], n_letras):
                    if plano_letras[c] not in " .":
                        sons.append((escritos / FPS, "tecla"))
                escrita["letras"] = n_letras
                if n_letras not in escrita["camadas"]:
                    escrita["camadas"] = {n_letras: camada_de_texto(p["texto"], fnt, y_centro=p.get("texto_y", 0.24),
                                                                    sombra=p.get("sombra", 0.55), letras=n_letras)}
                fr = por_cima(fr, escrita["camadas"][n_letras])
            elif texto is not None:
                entrada = p.get("texto_entrada", 5)
                fr = por_cima(fr, texto, min(1.0, (j + 1) / entrada) if entrada else 1.0)
            ultimo = fr
            escrever(com_logo(com_legenda(fr, escritos), escritos))
            escritos += 1

    # depois da última foto: o logo fica sobre a foto já escura, e o vídeo apaga-se no fim
    n_logo, n_fade = n_total - escritos, round(FADE_S * FPS)
    for j in range(n_logo):
        fr = com_logo(ultimo, escritos + j).astype(np.float32)
        if j >= n_logo - n_fade:
            fr *= (n_logo - j) / n_fade
        escrever(fr.astype(np.uint8))
    enc.stdin.close()
    enc.wait()

    if faixa:
        # uma pista com os efeitos: as teclas da máquina de escrever e os disparos de câmara
        efeitos = so_video.with_suffix(".efeitos.wav")
        pista, rng = np.zeros(int((total + 1) * SR)), np.random.default_rng(3)
        # o disparo: o do aluno, se o der; senão, o que o Sérgio escolheu (disparo.wav, ao lado do motor);
        # e só em último caso o gerado
        som_disparo = plano.get("som_disparo") or (Path(__file__).with_name("disparo.wav") if Path(__file__).with_name("disparo.wav").exists() else None)
        if som_disparo:
            # lê-se pelo ffmpeg, que aceita qualquer formato (há ficheiros .mp3 que são AAC por dentro)
            disparo = ler_audio(som_disparo)
            disparo = disparo / (np.max(np.abs(disparo)) + 1e-9) * 0.9
        else:
            disparo = disparo_amostras()
        vol_tecla, vol_disparo = plano.get("volume_teclas", 0.35), plano.get("volume_disparo", 1.4) * 0.7
        for t, tipo in sons:
            som = clique_de_maquina(rng) * vol_tecla if tipo == "tecla" else disparo * vol_disparo
            i = int(t * SR)
            pista[i:i + len(som)] += som[:max(0, len(pista) - i)]
        for t, parte in voz_partes:
            i = int(t * SR)
            pista[i:i + len(parte)] += parte[:max(0, len(pista) - i)] * plano["voz"].get("volume", 1.0)
        # a música, já cortada, com o volume a seguir a história
        musica = ler_audio(faixa, canais=2, inicio=t0, duracao=total)
        n = len(musica)
        tt = np.arange(n) / SR
        ganho = np.ones(n, np.float32)
        if revelar is not None:
            # antes do drop a música fica baixa, para a leitura e o clique das teclas; explode com a foto
            ganho[tt < cortes[revelar] / FPS] = plano.get("musica_baixa_antes_do_drop", 0.3)
        if voz_partes:
            # com a voz do cliente, a música baixa enquanto ela fala e sobe nas pausas e no fim
            baixo = plano["voz"].get("musica_por_baixo", 0.2)
            alto = plano["voz"].get("musica_nas_pausas", 0.85)
            alvo = np.full(n, alto, np.float32)
            for t, parte in voz_partes:
                alvo[(tt >= t - 0.15) & (tt < t + len(parte) / SR + 0.1)] = baixo
            # rampas de 0,25 s para não se ouvir o volume a saltar
            r = int(0.25 * SR)
            alvo = np.convolve(np.pad(alvo, (r // 2, r - r // 2 - 1), mode="edge"), np.ones(r) / r, "valid")
            ganho *= alvo.astype(np.float32)
        musica *= ganho[:, None]
        f_in, f_out = int(0.06 * SR), int(1.0 * SR)
        musica[:f_in] *= np.linspace(0, 1, f_in)[:, None]
        musica[-f_out:] *= np.linspace(1, 0, f_out)[:, None]
        mistura = musica + pista[:n, None]
        gravar_wav(efeitos, mistura)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(so_video), "-i", str(efeitos),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", str(saida)], check=True)
        efeitos.unlink()
        if plano.get("copia_sem_som"):
            subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(so_video), "-c", "copy",
                            str(saida.with_name(saida.stem + " (sem som).mp4"))], check=True)
        so_video.unlink()
        if revelar is not None:
            print(f"a foto revelada entra aos {cortes[revelar] / FPS:.2f}s do Reel, {t0 + cortes[revelar] / FPS:.2f}s da música")
        perto = [float(np.min(np.abs(ataques - (c / FPS + t0)))) for c in cortes[1:]]
        print(f"{sum(d < 0.06 for d in perto)}/{len(perto)} cortes a menos de 60 ms de um ataque da música")
    else:
        destino = saida.with_name(saida.stem + " (sem som).mp4")
        so_video.replace(destino)
        saida = destino
    print(f"exportado: {saida}")


if __name__ == "__main__":
    montar(sys.argv[1])

"""Funções partilhadas: abrir fotos, recortar e escolher o tipo de letra. Funciona no Mac e no Windows.

Regra da casa: as cores da foto são as do fotógrafo. Nada aqui altera cor, contraste
ou saturação. A única conversão de cor é a de gestão de cor: passar do perfil
embutido (Adobe RGB, Display P3, ...) para sRGB, para que a foto apareça no
Instagram exatamente como no ecrã do fotógrafo."""
import io
import os
from pathlib import Path

from PIL import Image, ImageCms, ImageFont, ImageOps

try:  # as fotos HEIC do iPhone; sem o pillow-heif, pede-se ao aluno que as exporte em JPG
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC = True
except ImportError:
    HEIC = False

EXTENSOES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif", ".webp",
             ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf"}
RAW = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf"}


def listar_fotos(pasta):
    """Devolve as fotos de uma pasta (recursivo), ordenadas por nome. Os RAW ficam de fora."""
    fotos = []
    for raiz, _, ficheiros in os.walk(os.path.expanduser(pasta)):
        for f in ficheiros:
            ext = os.path.splitext(f)[1].lower()
            if not f.startswith(".") and ext in EXTENSOES and ext not in RAW:
                fotos.append(os.path.join(raiz, f))
    return sorted(fotos)


def abrir(caminho, tamanho_max=None):
    """Abre uma foto já rodada segundo o EXIF, em RGB/sRGB.

    tamanho_max acelera a leitura de JPEGs grandes (miniaturas)."""
    ext = os.path.splitext(caminho)[1].lower()
    if ext in RAW:
        raise SystemExit(f"{caminho} é um RAW: não traz a edição do fotógrafo. "
                         "Exporta-o em JPG no Lightroom ou no Photoshop.")
    if ext in (".heic", ".heif") and not HEIC:
        raise SystemExit(f"{caminho} é HEIC e falta o pillow-heif. Exporta a foto em JPG.")
    im = Image.open(caminho)
    if tamanho_max and im.format == "JPEG":
        im.draft("RGB", (tamanho_max, tamanho_max))
    im = ImageOps.exif_transpose(im)
    return para_srgb(im)


SRGB = ImageCms.createProfile("sRGB")
SRGB_BYTES = ImageCms.ImageCmsProfile(SRGB).tobytes()


def para_srgb(im):
    """Converte para RGB/sRGB respeitando o perfil ICC embutido.

    Sem perfil, assume-se que a foto já é sRGB (é o que o Lightroom e o
    Photoshop exportam por defeito para web) e os valores não são tocados."""
    icc = im.info.get("icc_profile")
    if icc:
        origem = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        nome = ImageCms.getProfileDescription(origem).strip().lower()
        if "srgb" not in nome or im.mode not in ("RGB", "L"):
            im = ImageCms.profileToProfile(
                im, origem, SRGB,
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode="RGB",
                flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.info.pop("icc_profile", None)
    return im


def preencher(im, largura, altura, foco=(0.5, 0.45)):
    """Recorta a foto para preencher exatamente largura x altura (tipo 'cover').

    foco = (x, y) entre 0 e 1: o ponto da foto que deve ficar no centro do recorte.
    Por defeito ligeiramente acima do centro, onde costumam estar as caras."""
    escala = max(largura / im.width, altura / im.height)
    nova = im.resize((max(largura, round(im.width * escala)),
                      max(altura, round(im.height * escala))), Image.LANCZOS)
    fx, fy = foco
    x = min(max(round(nova.width * fx - largura / 2), 0), nova.width - largura)
    y = min(max(round(nova.height * fy - altura / 2), 0), nova.height - altura)
    return nova.crop((x, y, x + largura, y + altura))


def caber(im, largura, altura):
    """Redimensiona a foto para caber dentro da caixa, sem recortar (tipo 'contain')."""
    escala = min(largura / im.width, altura / im.height)
    return im.resize((round(im.width * escala), round(im.height * escala)), Image.LANCZOS)


# ---------- tipos de letra ----------
# Sem letra na ficha do aluno: a Bodoni e a Avenir Next no Mac; nos outros, a Bodoni Moda e a Nunito Sans
# que vêm com o motor (em letras/, licença OFL), e a DejaVu como último recurso.
LETRAS_DO_MOTOR = Path(__file__).resolve().parent / "letras"
LETRAS = {
    "titulo": [("/System/Library/Fonts/Supplemental/Bodoni 72.ttc", 2), (LETRAS_DO_MOTOR / "BodoniModa-Bold.ttf", 0),
               ("DejaVuSerif-Bold.ttf", 0)],
    "titulo_italico": [("/System/Library/Fonts/Supplemental/Bodoni 72.ttc", 1),
                       (LETRAS_DO_MOTOR / "BodoniModa-Italico.ttf", 0), ("DejaVuSerif-Italic.ttf", 0)],
    "texto": [("/System/Library/Fonts/Avenir Next.ttc", 7), (LETRAS_DO_MOTOR / "NunitoSans-Regular.ttf", 0),
              ("DejaVuSans.ttf", 0)],
}


def letra(papel, tamanho, escolhida=None):
    """papel: "titulo", "titulo_italico" ou "texto". escolhida: [caminho, índice] ou caminho, da ficha do aluno."""
    candidatos = []
    if escolhida:
        candidatos.append(tuple(escolhida) if isinstance(escolhida, (list, tuple)) else (escolhida, 0))
    for c, i in candidatos + LETRAS[papel]:
        try:
            return ImageFont.truetype(str(Path(c).expanduser()), tamanho, index=i)
        except OSError:
            continue
    return ImageFont.load_default(tamanho)

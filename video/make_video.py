# Baba'ya sevgi videosu: fotoğraftan Ken Burns sahneleri + yazılar + üretilmiş müzik -> MP4
# Kullanım: python3 make_video.py <foto.jpg> <cikti.mp4>
import subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, OUT = sys.argv[1], sys.argv[2]
W, H, FPS, SCENE, XF = 1280, 720, 25, 10.0, 1.0
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

img = Image.open(SRC).convert("RGB")
if img.size == (1080, 2640):  # gönderilen telefon ekran görüntüsü: siyah bantları kırp
    img = img.crop((0, 915, 1080, 1725))

def box(x, y, w):
    return (x, y, w, w * 9 / 16)

# (başlangıç kutusu, bitiş kutusu, başlık, alt yazı)
SCENES = [
    (box(0, 0, 1080), box(60, 30, 960), "Bir yaz günü...", "Ege'nin mavisinde, en değerli hazinemizle"),
    (box(60, 120, 960), box(0, 100, 1080), "Biz bir aileyiz", "Ve bu ailenin kalbi: Babamız"),
    (box(560, 200, 360), box(590, 230, 300), "Babacığım", "Yüzündeki o gülümseme bizim güneşimiz"),
    (box(470, 200, 600), box(440, 170, 640), "Güçlü kolların", "Bizim en güvenli limanımız"),
    (box(440, 390, 280), box(410, 370, 330), "En küçük prensesin", "Henüz konuşamıyor ama kalbi seni çok seviyor"),
    (box(200, 520, 330), box(220, 545, 280), "Bir prensesin", "Her sabah seni görmek için uyanıyor"),
    (box(90, 560, 320), box(60, 520, 380), "Bir prensesin daha", "Senin gibi cesur ve güçlü olmak istiyor"),
    (box(230, 340, 290), box(190, 310, 360), "Ve hayat arkadaşın", "Bu yolda elini hiç bırakmayan"),
    (box(120, 300, 640), box(150, 330, 560), "Üç kız, bir anne", "Hepimizin kalbi seninle atıyor"),
    (box(480, 180, 560), box(420, 140, 660), "Teşekkürler baba", "Sevgin, sabrın ve emeğin için"),
    (box(150, 160, 860), box(0, 100, 1080), "İyi ki varsın", "İyi ki bizim babamızsın"),
    (box(0, 100, 1080), box(0, 0, 1080), "Seni çok seviyoruz ♥", "Eşin ve kızların"),
]
TOTAL = len(SCENES) * SCENE + XF

f_title = ImageFont.truetype(FONT_B, 60)
f_sub = ImageFont.truetype(FONT_R, 34)

def ease(t):
    return t * t * (3 - 2 * t)

def text_layer(title, sub):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rectangle((0, H - 200, W, H), fill=(0, 0, 0, 110))
    for text, font, y in ((title, f_title, H - 175), (sub, f_sub, H - 92)):
        tw = d.textlength(text, font=font)
        x = (W - tw) / 2
        d.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0, 200))
        d.text((x, y), text, font=font, fill=(255, 245, 230, 255))
    return layer.filter(ImageFilter.SMOOTH)

LAYERS = [text_layer(s[2], s[3]) for s in SCENES]

def scene_frame(i, t):  # t: sahne içindeki saniye, 0..SCENE+XF
    a, b, _, _ = SCENES[i]
    k = ease(min(t / (SCENE + XF), 1))
    x, y, w, h = (a[j] + (b[j] - a[j]) * k for j in range(4))
    fr = img.resize((W, H), Image.BICUBIC, box=(x, y, x + w, y + h)).convert("RGBA")
    alpha = min(1, max(0, (t - 0.8) / 0.8), max(0, (SCENE - 0.3 - t) / 0.7))  # yazı belirip kaybolur
    if alpha > 0:
        lay = LAYERS[i].copy()
        lay.putalpha(lay.getchannel("A").point(lambda v: int(v * alpha)))
        fr = Image.alpha_composite(fr, lay)
    return fr.convert("RGB")

def music(path):
    sr = 44100
    n = int(TOTAL * sr)
    out = np.zeros(n + sr * 12)
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    prog = [[48, 55, 64, 67, 72], [43, 55, 62, 67, 71], [45, 57, 64, 69, 72], [41, 53, 60, 65, 69]]  # C G Am F
    beat = 60 / 72
    t_chord = 4 * beat * 2
    c = 0
    while c * t_chord < TOTAL:
        notes = prog[c % 4]
        s0 = int(c * t_chord * sr)
        L = int(t_chord * sr)
        tt = np.arange(L) / sr
        env = np.minimum(1, tt / 1.5) * np.minimum(1, (t_chord - tt) / 1.5)
        for m in notes[:3]:  # yumuşak pad
            out[s0:s0 + L] += 0.05 * env * (np.sin(2 * np.pi * hz(m) * tt) + 0.3 * np.sin(4 * np.pi * hz(m) * tt))
        for k in range(8):  # piyano benzeri arpej
            m = notes[[1, 2, 3, 4, 3, 2, 3, 4][k]] + 12
            s = s0 + int(k * beat * sr)
            ln = int(3 * sr)
            tn = np.arange(ln) / sr
            tone = sum(a * np.sin(2 * np.pi * hz(m) * h * tn) for h, a in ((1, 1), (2, 0.4), (3, 0.15)))
            out[s:s + ln] += 0.12 * np.exp(-tn * 2.2) * np.minimum(1, tn / 0.01) * tone
        c += 1
    out = out[:n]
    d = int(0.33 * sr)
    for _ in range(3):  # basit eko
        out[d:] += 0.35 * out[:-d]
    fade = np.ones(n)
    fade[: 2 * sr] = np.linspace(0, 1, 2 * sr)
    fade[-4 * sr:] = np.linspace(1, 0, 4 * sr)
    out = out * fade / np.abs(out).max() * 0.8
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((out * 32767).astype(np.int16).tobytes())

wav = OUT + ".wav"
music(wav)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
                       "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", OUT],
                      stdin=subprocess.PIPE)
for f in range(int(TOTAL * FPS)):
    t = f / FPS
    i = min(int(t // SCENE), len(SCENES) - 1)
    ts = t - i * SCENE
    fr = scene_frame(i, ts)
    if i > 0 and ts < XF:  # önceki sahneden geçiş
        fr = Image.blend(scene_frame(i - 1, ts + SCENE), fr, ease(ts / XF))
    if t < 1.5:
        fr = Image.blend(Image.new("RGB", (W, H)), fr, t / 1.5)
    if t > TOTAL - 2.5:
        fr = Image.blend(fr, Image.new("RGB", (W, H)), (t - (TOTAL - 2.5)) / 2.5)
    ff.stdin.write(fr.tobytes())
ff.stdin.close()
assert ff.wait() == 0, "ffmpeg hata verdi"

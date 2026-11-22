# Parkta Bir Gün: baba 2 kızıyla salıncak/kaydırakta oynar, anne bankta bebekle izler. 60 sn çizgi film -> MP4
# Kullanım: python3 park_film.py <cikti.mp4>
import math, os, subprocess, sys, wave
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
W, H, FPS, DUR, S = 1280, 720, 25, 60.0, 2  # S: kenar yumuşatma için 2x çizim
GROUND = 610
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

def vec(a):  # 0 = aşağı, 90 = sağ, 180 = yukarı, -90 = sol
    r = math.radians(a); return (math.sin(r), math.cos(r))
def add(p, v, l=1): return (p[0] + v[0] * l, p[1] + v[1] * l)
def lerp(a, b, k): return a + (b - a) * k
def ease(k): k = max(0.0, min(1.0, k)); return k * k * (3 - 2 * k)

class Pen:
    def __init__(s, img): s.d = ImageDraw.Draw(img)
    def P(s, p): return (p[0] * S, p[1] * S)
    def B(s, b): return [v * S for v in b]
    def circle(s, c, r, col): s.d.ellipse(s.B((c[0] - r, c[1] - r, c[0] + r, c[1] + r)), fill=col)
    def line(s, pts, col, w):
        s.d.line([s.P(p) for p in pts], fill=col, width=max(1, int(w * S)), joint="curve")
        for p in (pts[0], pts[-1]): s.circle(p, w / 2, col)
    def ell(s, b, col): s.d.ellipse(s.B(b), fill=col)
    def rect(s, b, col): s.d.rectangle(s.B(b), fill=col)
    def poly(s, pts, col): s.d.polygon([s.P(p) for p in pts], fill=col)
    def pie(s, b, a0, a1, col): s.d.pieslice(s.B(b), a0, a1, fill=col)
    def arc(s, b, a0, a1, col, w): s.d.arc(s.B(b), a0, a1, fill=col, width=max(1, int(w * S)))

# ---------- karakterler (fotoğraftaki kıyafetler) ----------
CH = {
    "dad": dict(h=250, kid=False, skin=(222, 172, 138), shirt=(30, 45, 95), bottom=(185, 165, 125), shoe=(70, 60, 55), hair="dad"),
    "mom": dict(h=228, kid=False, skin=(242, 205, 178), shirt=(28, 28, 34), bottom=(28, 28, 34), shoe=(120, 70, 60), hair="mom"),
    "g1": dict(h=138, kid=True, skin=(240, 200, 170), shirt=(40, 70, 175), bottom=(240, 240, 240), shoe=(230, 90, 120), hair="g1"),
    "g2": dict(h=122, kid=True, skin=(240, 202, 172), shirt=(175, 225, 70), bottom=(250, 170, 195), shoe=(245, 245, 245), hair="g2"),
    "baby": dict(h=58, kid=True, skin=(245, 210, 185), shirt=(252, 252, 252), bottom=(215, 40, 55), shoe=(215, 40, 55), hair="baby"),
}
def dims(key):
    h = CH[key]["h"]
    if key == "baby": return dict(hr=.2 * h, T=.3 * h, th=.17 * h, sh=.15 * h, ua=.15 * h, fa=.13 * h, lw=.09 * h, tw=.26 * h)
    if CH[key]["kid"]: return dict(hr=.15 * h, T=.3 * h, th=.2 * h, sh=.2 * h, ua=.15 * h, fa=.14 * h, lw=.075 * h, tw=.2 * h)
    return dict(hr=.1 * h, T=.33 * h, th=.24 * h, sh=.24 * h, ua=.17 * h, fa=.16 * h, lw=.055 * h, tw=.16 * h)
def hip_stand(key): d = dims(key); return GROUND - d["th"] - d["sh"] - d["lw"] * .5
def shoulder(key, hip, r=0): return add(hip, vec(r + 180), dims(key)["T"] * .9)

def ik(a, t, l1, l2, bend):
    dx, dy = t[0] - a[0], t[1] - a[1]; d = math.hypot(dx, dy) or 1e-3
    dd = max(abs(l1 - l2) + .1, min(d, l1 + l2 - .1)); ux, uy = dx / d, dy / d
    al = math.acos(max(-1, min(1, (l1 * l1 + dd * dd - l2 * l2) / (2 * l1 * dd)))) * bend
    ca, sa = math.cos(al), math.sin(al)
    return (a[0] + l1 * (ux * ca - uy * sa), a[1] + l1 * (ux * sa + uy * ca)), (a[0] + ux * dd, a[1] + uy * dd)

STAND = ((0, 0), (0, 0))
REST = (("ang", -15, -8), ("ang", 15, 8))
def SIT(f): return ((88 * f, 0), (80 * f, 4 * f))
def walk(p, f, A=28):
    s, c = math.sin(p), math.cos(p)
    legs = ((A * s * f, A * s * f - f * 20 * max(0, c)), (-A * s * f, -A * s * f - f * 20 * max(0, -c)))
    arms = (("ang", -.8 * A * s * f, -.8 * A * s * f + 25 * f), ("ang", .8 * A * s * f, .8 * A * s * f + 25 * f))
    return legs, arms

def hair_back(pen, hk, x, y, r):
    if hk == "mom":
        pen.ell((x - 1.3 * r, y + .6 * r, x + 1.3 * r, y + 2.7 * r), (175, 125, 75))  # karamel uçlar
        pen.ell((x - 1.3 * r, y - 1.1 * r, x + 1.3 * r, y + 1.9 * r), (75, 48, 32))
    elif hk in ("g1", "g2"):
        pen.ell((x - 1.1 * r, y - .8 * r, x + 1.1 * r, y + 1.5 * r), (65, 42, 30))

def head(pen, hk, skin, x, y, r, f, mouth):
    pen.circle((x, y), r, skin)
    ex = f * .25 * r
    if hk == "dad":
        pen.pie((x - r, y - r, x + r, y + r), 12, 168, (65, 50, 42))  # sakal
        pen.pie((x - 1.05 * r, y - 1.2 * r, x + 1.05 * r, y + .5 * r), 180, 360, (40, 32, 28))
        for s in (-1, 1): pen.line([(x + ex + s * .2 * r, y - .3 * r), (x + ex + s * .5 * r, y - .33 * r)], (40, 32, 28), .09 * r)
    elif hk == "mom":
        pen.pie((x - 1.08 * r, y - 1.15 * r, x + 1.08 * r, y + .7 * r), 180, 360, (75, 48, 32))
        for s in (-1, 1): pen.circle((x + s * .38 * r, y - 1.0 * r), .24 * r, (20, 20, 25))  # başta güneş gözlüğü
    elif hk in ("g1", "g2"):
        hat = (125, 175, 220) if hk == "g1" else (240, 215, 190)
        pen.pie((x - 1.05 * r, y - 1.45 * r, x + 1.05 * r, y + .05 * r), 180, 360, hat)
        if hk == "g2": pen.line([(x - .9 * r, y - .75 * r), (x + .9 * r, y - .75 * r)], (220, 70, 80), .12 * r)
        pen.ell((x - 1.5 * r, y - .8 * r, x + 1.5 * r, y - .5 * r), hat)
    elif hk == "baby":
        for s in (-.3, 0, .3): pen.arc((x + s * r - .25 * r, y - 1.15 * r, x + s * r + .25 * r, y - .75 * r), 180, 330, (150, 110, 70), .07 * r)
    ey = -.12 * r if hk == "dad" else 0
    if hk != "dad":
        for s in (-1, 1): pen.circle((x + ex + s * .55 * r, y + .3 * r), .13 * r, (250, 170, 165))
    for s in (-1, 1): pen.circle((x + ex + s * .33 * r, y + ey), .1 * r, (30, 25, 25))
    mx = x + ex * .8
    if hk == "dad":
        pen.pie((mx - .38 * r, y + .02 * r, mx + .38 * r, y + (.6 if mouth == "open" else .45) * r), 0, 180, (250, 250, 250))
    elif mouth == "open":
        pen.ell((mx - .2 * r, y + .25 * r, mx + .2 * r, y + .62 * r), (150, 40, 50))
    else:
        pen.arc((mx - .33 * r, y + .05 * r, mx + .33 * r, y + .5 * r), 20, 160, (120, 40, 40), max(.6, .07 * r))

def person(pen, key, hip, r=0, f=1, legs=STAND, arms=REST, mouth="smile", parts="all"):
    c, d = CH[key], dims(key)
    up = vec(r + 180)
    sh = add(hip, up, d["T"] * .9)
    hx, hy = add(hip, up, d["T"] + d["hr"] * .85)
    if parts in ("all", "body"):
        hair_back(pen, c["hair"], hx, hy, d["hr"])
        for a1, a2 in legs:
            knee = add(hip, vec(a1 + r), d["th"]); foot = add(knee, vec(a2 + r), d["sh"])
            pen.line([knee, foot], c["skin"] if key != "baby" else c["bottom"], d["lw"])
            pen.line([hip, knee], c["bottom"], d["lw"] * 1.3)
            pen.circle(add(foot, (f * d["lw"] * .4, 0)), d["lw"] * .7, c["shoe"])
        pen.circle(hip, d["tw"] * .45, c["bottom"])
        pen.line([add(hip, up, d["tw"] * .3), sh], c["shirt"], d["tw"])
        if key == "dad" and abs(r) < 30:
            m = add(hip, up, d["T"] * .5)
            pen.d.text(pen.P(m), "35", fill=(200, 40, 60), anchor="mm", font=FONT35)
        head(pen, c["hair"], c["skin"], hx, hy, d["hr"], f, mouth)
    if parts in ("all", "arms"):
        for a in arms:
            if a[0] == "ang":
                el = add(sh, vec(a[1] + r), d["ua"]); hd = add(el, vec(a[2] + r), d["fa"])
            else:
                el, hd = ik(sh, (a[1], a[2]), d["ua"], d["fa"], a[3])
            pen.line([sh, el], c["skin"], d["lw"])
            pen.line([sh, ((sh[0] + el[0]) / 2, (sh[1] + el[1]) / 2)], c["shirt"], d["lw"] * 1.35)
            pen.line([el, hd], c["skin"], d["lw"])
            pen.circle(hd, d["lw"] * .65, c["skin"])
    return sh

FONT35 = ImageFont.truetype(FONT_B, int(12 * S))

# ---------- dekor ----------
P1, P2, ROPE = (490, 335), (650, 335), 185
def chute_y(x):  # kaydırak yüzeyi
    return (405 + (x - 955) * 175 / 195 if x < 1150 else 580 + (x - 1150) * .2) - 8

def tree(pen, x, yb, s):
    pen.rect((x - 8 * s, yb - 120 * s, x + 8 * s, yb), (110, 75, 45))
    for dx, dy, r, col in ((-35, -140, 45, (60, 130, 55)), (30, -150, 48, (70, 145, 60)), (0, -190, 52, (80, 160, 65)), (-10, -130, 40, (90, 170, 70))):
        pen.circle((x + dx * s, yb + dy * s), r * s, col)

def background():
    img = Image.new("RGB", (W * S, H * S)); pen = Pen(img)
    for y in range(0, 335):
        k = y / 335; pen.rect((0, y, W, y + 1), tuple(int(lerp(a, b, k)) for a, b in zip((110, 170, 230), (205, 232, 250))))
    pen.circle((1120, 95), 62, (255, 242, 180)); pen.circle((1120, 95), 45, (255, 225, 110))
    pen.rect((0, 330, W, 375), (40, 110, 170))  # deniz
    pen.pie((120, 318, 520, 342), 180, 360, (110, 125, 85)); pen.pie((880, 322, 1060, 338), 180, 360, (120, 130, 90))
    pen.rect((0, 372, W, H), (125, 190, 85))
    for i in range(0, W, 26): pen.circle((i + 8, 380 + (i * 7) % 9), 15, (60, 120, 55))
    pen.ell((360, 555, 1290, 690), (226, 206, 160))  # oyun alanı kumu
    rnd = np.random.default_rng(3)
    for _ in range(140):
        x, y = rnd.uniform(0, W), rnd.uniform(395, H)
        if not (360 < x < 1290 and 555 < y < 690):
            pen.circle((x, y), 3, [(255, 255, 255), (250, 210, 60), (240, 120, 160)][int(rnd.integers(3))])
    tree(pen, 45, 600, 1.25); tree(pen, 790, 545, .75); tree(pen, 1250, 560, .9)
    # salıncak
    for top, legs in (((420, 330), (395, 445)), ((720, 330), (695, 745))):
        for lx in legs: pen.line([(lx, GROUND + 2), top], (210, 60, 50), 8)
    pen.line([(410, 330), (730, 330)], (240, 190, 40), 10)
    # kaydırak
    for x in (860, 895): pen.line([(x, GROUND + 2), (x, 395)], (90, 140, 200), 6)
    for y in range(588, 400, -30): pen.line([(860, y), (895, y)], (90, 140, 200), 4)
    pen.line([(1085, 530), (1085, GROUND + 2)], (150, 150, 160), 7)
    pen.line([(965, 400), (965, GROUND + 2)], (90, 140, 200), 6)
    pen.rect((855, 398, 968, 412), (90, 140, 200))
    pen.line([(858, 398), (858, 360), (965, 360), (965, 398)], (90, 140, 200), 4)
    pen.line([(955, 405), (1150, 580), (1200, 590)], (245, 150, 40), 16)
    pen.line([(955, 396), (1150, 571), (1200, 581)], (215, 115, 25), 5)
    # bank
    for x in (110, 310): pen.line([(x, 552), (x, GROUND + 2)], (80, 60, 45), 7)
    for y in (498, 520): pen.rect((95, y, 325, y + 12), (160, 110, 60))
    for x in (105, 315): pen.line([(x, 495), (x, 552)], (80, 60, 45), 6)
    pen.rect((92, 550, 328, 562), (170, 118, 65))
    return img

def cloud(pen, x, y, s):
    for dx, dy, r in ((0, 0, 28), (30, -12, 32), (60, 0, 26), (30, 8, 26)): pen.circle((x + dx * s, y + dy * s), r * s, (255, 255, 255))

def heart(pen, x, y, s, col):
    pen.circle((x - s * .5, y), s * .55, col); pen.circle((x + s * .5, y), s * .55, col)
    pen.poly([(x - s * 1.03, y + s * .15), (x + s * 1.03, y + s * .15), (x, y + s * 1.25)], col)

def draw_swing(pen, a1, a2):
    seats = []
    for p, a in ((P1, a1), (P2, a2)):
        s = add(p, vec(a), ROPE); per = (math.cos(math.radians(a)), -math.sin(math.radians(a)))
        pen.line([p, s], (80, 70, 60), 2.5)
        pen.line([add(s, per, -20), add(s, per, 20)], (120, 80, 50), 6)
        seats.append(s)
    return seats

def draw_mom(pen, t, mouth="smile", excited=False):
    hip = (160, 548)
    person(pen, "mom", hip, f=1, legs=SIT(1), mouth=mouth, parts="body")
    bw = math.sin(t * (9 if excited else 4))
    person(pen, "baby", (196, 534), f=1, legs=((80, 0), (70, 10)),
           arms=(("ang", 150 + 30 * bw, 165 + 30 * bw), ("ang", 55, 80)), mouth="open" if excited else "smile")
    person(pen, "mom", hip, f=1, arms=(("to", 186, 522, 1), ("to", 212, 524, 1)), parts="arms")

# ---------- sahneler: her biri çizer ve (kamera, yazı) döndürür ----------
def sc_intro(pen, t):  # 0-6: baba kızlarla el ele parka gelir
    draw_mom(pen, t)
    draw_swing(pen, 3 * math.sin(t), -2 * math.sin(t))
    x = lerp(1330, 640, ease(t / 5.5)); moving = t < 5.4
    p = x / 11
    dl, da = walk(p, -1) if moving else (STAND, REST)
    dh = (x, hip_stand("dad") - (3 * abs(math.sin(p)) if moving else 0)); dsh = shoulder("dad", dh)
    _, hA = ik(dsh, (x - 40, dh[1] + 10), dims("dad")["ua"], dims("dad")["fa"], -1)
    _, hB = ik(dsh, (x + 40, dh[1] + 10), dims("dad")["ua"], dims("dad")["fa"], 1)
    person(pen, "dad", dh, f=-1, legs=dl, arms=(("to", *hA, -1), ("to", *hB, 1)), mouth="smile")
    for key, gx, hold in (("g1", x - 72, hA), ("g2", x + 68, hB)):
        gl, ga = walk(p + 1.3, -1, 32) if moving else (STAND, REST)
        person(pen, key, (gx, hip_stand(key) - (3 * abs(math.sin(p + 1.3)) if moving else 0)), f=-1, legs=gl,
               arms=(("to", *hold, 1), ga[1]), mouth="open" if moving else "smile")
    z = lerp(1.0, 1.15, ease(t / 6))
    return (lerp(640, 660, t / 6), lerp(360, 400, t / 6), z), ("title", t, 0.6, 5.6)

def sc_swing(pen, t, u):  # 6-20: baba iki kızı aynı anda sallar; 16'da annenin yakın çekimi
    w = 2 * math.pi / 2.6
    a1 = 22 * math.sin(w * u); a2 = -a1; vel = math.cos(w * u)
    close = u >= 10
    draw_mom(pen, t, mouth="open" if close else "smile", excited=close)
    s1 = add(P1, vec(a1), ROPE); s2 = add(P2, vec(a2), ROPE)
    backs = []
    for key, pv, a, f, v in (("g1", P1, a1, -1, vel), ("g2", P2, a2, 1, -vel)):
        d = dims(key); seat = add(pv, vec(a), ROPE)
        hip = add(seat, vec(a + 180), d["lw"] * .8)
        backs.append(add(add(hip, vec(a + 180), d["T"] * .55), (-f * d["tw"] * .5, 0)))
    dh = (570, hip_stand("dad"))
    person(pen, "dad", dh, f=0, legs=((-6, -6), (6, 6)), arms=(("to", *backs[0], -1), ("to", *backs[1], 1)), mouth="open")
    draw_swing(pen, a1, a2)
    for key, pv, a, f, v in (("g1", P1, a1, -1, vel), ("g2", P2, a2, 1, -vel)):
        d = dims(key); seat = add(pv, vec(a), ROPE)
        hip = add(seat, vec(a + 180), d["lw"] * .8)
        k = .5 + .5 * v * f if f == -1 else .5 + .5 * v
        k = max(0, min(1, k))
        hand = add(pv, vec(a), ROPE - d["T"] * .8)
        person(pen, key, hip, r=a, f=f, legs=((85 * f, f * (5 + 70 * k)), (80 * f, f * (10 + 70 * k))),
               arms=(("to", *hand, -f), ("to", *hand, -f)), mouth="open" if k > .55 else "smile")
    if close:
        return (200, 470, 2.3), None
    return (570, 455, lerp(1.55, 1.75, u / 10)), None

def slide_kid(pen, key, u):
    d = dims(key); u %= 6; hs = hip_stand(key)
    if u < 1.6:
        x = lerp(1190, 878, u / 1.6); p = u * 13; legs, arms = walk(p, -1, 35)
        person(pen, key, (x, hs - 3 * abs(math.sin(p))), f=-1, legs=legs, arms=arms, mouth="open")
    elif u < 3.4:
        k = (u - 1.6) / 1.8; feet = lerp(612, 425, k); hy = feet - (d["th"] + d["sh"]) * .85; s = math.sin(k * 18)
        ry = hy - d["T"] * 1.05
        person(pen, key, (882, hy), f=-1, legs=((-25 - 25 * s, 10), (-25 + 25 * s, 10)),
               arms=(("to", 862, ry - 8 * s, 1), ("to", 862, ry + 8 * s, 1)))
    elif u < 3.9:
        x = lerp(885, 955, (u - 3.4) / .5)
        person(pen, key, (x, 398 - d["lw"] * .7), f=1, legs=SIT(1), arms=(("ang", 40, 60), ("ang", 50, 70)))
    elif u < 5.1:
        x = lerp(955, 1185, ((u - 3.9) / 1.2) ** 1.6)
        legs = ((40, 40), (45, 45)) if x < 1150 else ((75, 75), (80, 80))
        person(pen, key, (x, chute_y(x) - d["lw"] * .6), r=15, f=1, legs=legs,
               arms=(("ang", 165, 170), ("ang", 150, 160)), mouth="open")
    else:
        k = (u - 5.1) / .9; x = lerp(1185, 1212, k); y = hs - 45 * math.sin(math.pi * k)
        person(pen, key, (x, y), f=1, arms=(("ang", 150, 165), ("ang", 165, 175)), mouth="open")
        return (x, y)

def sc_slide(pen, t, u):  # 20-34
    draw_mom(pen, t)
    draw_swing(pen, 3 * math.sin(t), -2 * math.sin(t))
    dh = (1245, hip_stand("dad"))
    # babayı önce çiz; atlayan kızı yakalaması için kolları sonra
    person(pen, "dad", dh, f=-1, mouth="open", parts="body")
    jump = None
    for key, off in (("g1", 0), ("g2", 3)):
        j = slide_kid(pen, key, u + off)
        jump = jump or (j and (key, j))
    if jump:
        key, (x, y) = jump; d = dims(key)
        arms = (("to", x + d["tw"] * .5, y - d["T"] * .4, 1), ("to", x + d["tw"] * .5, y - d["T"] * .75, 1))
    else:
        arms = (("ang", -150, -170), ("ang", -160, -175))
    person(pen, "dad", dh, f=-1, arms=arms, parts="arms")
    return (lerp(1030, 1040, u / 14), lerp(480, 470, u / 14), lerp(1.45, 1.6, u / 14)), None

def sc_plane(pen, t, u):  # 34-46: baba küçük kızı uçak gibi uçurur, büyük kız etrafında koşar
    draw_mom(pen, t, mouth="open", excited=True)
    draw_swing(pen, 3 * math.sin(t), -2 * math.sin(t))
    dx = 760 + 60 * math.sin(.6 * u)
    dh = (dx, hip_stand("dad")); dsh = shoulder("dad", dh)
    gx, gy = dx + 5, dsh[1] - 70 + 12 * math.sin(2.2 * u)
    rr = -85 + 10 * math.sin(1.1 * u); d2 = dims("g2")
    ghip = (gx - d2["T"] * .5, gy)
    person(pen, "dad", dh, f=0, legs=((-8, -8), (8, 8)),
           arms=(("to", gx - 14, gy + d2["tw"] * .5, -1), ("to", gx + 16, gy + d2["tw"] * .5, 1)), mouth="open")
    person(pen, "g2", ghip, r=rr, f=1, legs=((0, 0), (8, 8)), arms=(("ang", 175, 178), ("ang", 182, 180)), mouth="open")
    ph = .9 * u; x1 = 760 + 175 * math.sin(ph); f1 = 1 if math.cos(ph) > 0 else -1
    legs, _ = walk(u * 11, f1, 35)
    person(pen, "g1", (x1, hip_stand("g1") + 12 - 3 * abs(math.sin(u * 11))), f=f1, legs=legs,
           arms=(("ang", -85, -85), ("ang", 85, 85)), mouth="open")
    for i in range(6):  # kalpler
        k = (u * .5 + i / 6) % 1
        heart(pen, 200 + 30 * math.sin(i * 2 + u), 500 - 160 * k, 7 + 4 * (1 - k), (235, 70, 110))
    if u < 6:
        return (lerp(760, 740, u / 6), 430, 1.7), None
    k = ease((u - 6) / 6)
    return (lerp(740, 640, k), lerp(430, 360, k), lerp(1.7, 1.0, k)), None

def sc_bench(pen, t, u):  # 46-60: herkes bankta buluşur
    draw_swing(pen, 3 * math.sin(t), -2 * math.sin(t))
    k = ease(u / 4); moving = u < 4
    sit = ease((u - 4) / .6)
    dx = lerp(760, 262, k)
    dhy = lerp(hip_stand("dad"), 548, sit)
    p = dx / 11
    if moving:
        dl, _ = walk(p, -1)
    else:
        dl = tuple((lerp(0, a, sit), lerp(0, b, sit)) for a, b in SIT(1))
    dh = (dx, dhy - (3 * abs(math.sin(p)) if moving else 0))
    if not moving:  # sarılan kol annenin arkasından geçsin
        person(pen, "dad", dh, f=1, arms=(("to", 150, 478, 1),), parts="arms")
    draw_mom(pen, t, mouth="open" if u > 5 else "smile", excited=u > 5)
    person(pen, "dad", dh, f=-1 if moving else 1, legs=dl, mouth="open" if u > 5 else "smile", parts="body")
    # küçük kız babanın kucağına
    lap = ease((u - 4.4) / .6); d2 = dims("g2")
    g2x = lerp(lerp(680, 300, k), 300, lap)
    g2y = lerp(hip_stand("g2"), 535, lap) - 40 * math.sin(math.pi * lap)
    gl = walk(p + 1.3, -1, 32)[0] if moving else (STAND if lap < .5 else SIT(1))
    person(pen, "g2", (g2x, g2y), f=-1 if moving else 1, legs=gl, arms=(("ang", 150, 160), ("ang", 40, 70)) if u > 5 else REST,
           mouth="open" if u > 5 else "smile")
    g1x = lerp(860, 370, k)
    if moving:
        gl, ga = walk(p + 2, -1, 32)
    else:
        wv = math.sin(u * 8)
        gl, ga = STAND, (("ang", -150 + 20 * wv, -165 + 20 * wv), ("to", 330, 520, 1))
    person(pen, "g1", (g1x, hip_stand("g1")), f=-1, legs=gl, arms=ga, mouth="open" if u > 5 else "smile")
    if moving:
        darms = (("ang", 15, 10), ("ang", -15, -10))
    else:
        darms = (("to", g2x + 10, g2y - d2["T"] * .5, 1),)
    person(pen, "dad", dh, f=-1 if moving else 1, arms=darms, parts="arms")
    if u > 4.5:
        for i in range(12):
            st = 4.5 + i * .55
            if u > st:
                k2 = (u - st) / 4
                if k2 < 1:
                    heart(pen, 120 + i * 23 + 15 * math.sin(u * 2 + i), 470 - 150 * k2, 6 + 5 * (1 - k2),
                          [(235, 70, 110), (250, 120, 150), (220, 40, 70)][i % 3])
    k = ease((u - 3) / 5)
    return (lerp(640, 240, k), lerp(360, 490, k), lerp(1.0, 1.9, k)), ("end", u, 6, 14)

def scene(pen, t):
    if t < 6: return sc_intro(pen, t)
    if t < 20: return sc_swing(pen, t, t - 6)
    if t < 34: return sc_slide(pen, t, t - 20)
    if t < 46: return sc_plane(pen, t, t - 34)
    return sc_bench(pen, t, t - 46)
CUTS = [6, 16, 20, 34, 46]

# ---------- yazılar ----------
def text_layer(title, sub):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for txt, font, y in ((title, ImageFont.truetype(FONT_B, 58), 75), (sub, ImageFont.truetype(FONT_R, 32), 150)):
        tw = d.textlength(txt, font=font); x = (W - tw) / 2
        d.text((x + 3, y + 3), txt, font=font, fill=(0, 0, 0, 170))
        d.text((x, y), txt, font=font, fill=(255, 248, 235, 255))
    return lay
TEXTS = {"title": text_layer("Parkta Bir Gün", "Babamız ve kızları"),
         "end": text_layer("Seni çok seviyoruz Babacığım ♥", "Eşin ve kızların")}

BG = None
def render(fi):
    global BG
    if BG is None: BG = background()
    t = fi / FPS
    img = BG.copy(); pen = Pen(img)
    for x0, y, s, sp in ((100, 90, 1.0, 12), (600, 60, .8, 8), (1000, 140, 1.2, 10), (1400, 110, .9, 14)):
        cloud(pen, (x0 + sp * t) % 1600 - 160, y, s)
    for i in range(3):  # kuşlar
        bx = (300 + 40 * t + i * 35) % 1500 - 100; by = 150 + i * 12 + 6 * math.sin(t * 3 + i)
        wv = 5 * math.sin(t * 10 + i)
        pen.line([(bx - 8, by - wv), (bx, by)], (60, 60, 70), 2); pen.line([(bx, by), (bx + 8, by - wv)], (60, 60, 70), 2)
    (cx, cy, z), txt = scene(pen, t)
    cw, chh = W / z, H / z
    x0 = max(0, min(W - cw, cx - cw / 2)); y0 = max(0, min(H - chh, cy - chh / 2))
    fr = img.resize((W, H), Image.BICUBIC, box=(x0 * S, y0 * S, (x0 + cw) * S, (y0 + chh) * S))
    if t > 49:  # gün batımı
        fr = Image.blend(fr, Image.new("RGB", (W, H), (255, 150, 80)), .22 * min(1, (t - 49) / 8))
    if txt:
        name, u, a, b = txt
        al = min(1, max(0, (u - a) / .8), max(0, (b - u) / .8))
        if al > 0:
            lay = TEXTS[name].copy(); lay.putalpha(lay.getchannel("A").point(lambda v: int(v * al)))
            fr = Image.alpha_composite(fr.convert("RGBA"), lay).convert("RGB")
    dark = min([1.0, t / 1.0, (DUR - t) / 2.0] + [abs(t - c) / .2 for c in CUTS])
    if dark < 1: fr = Image.blend(Image.new("RGB", (W, H)), fr, max(0, dark))
    ImageDraw.Draw(fr).rectangle((0, 0, W, 44), fill=0); ImageDraw.Draw(fr).rectangle((0, H - 44, W, H), fill=0)  # sinema bantları
    return fr.tobytes()

def music(path):
    sr = 44100; n = int(DUR * sr); out = np.zeros(n + sr * 3)
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    def pluck(m, s, amp, dec, ln):
        L = int(ln * sr); tn = np.arange(L) / sr; i = int(s * sr)
        tone = sum(a * np.sin(2 * np.pi * hz(m) * h * tn) for h, a in ((1, 1), (2, .5), (3, .25), (4, .1)))
        seg = amp * np.exp(-tn * dec) * np.minimum(1, tn / .005) * tone
        out[i:i + L] += seg[:len(out) - i]
    prog = [[60, 64, 67, 72], [55, 59, 62, 67], [57, 60, 64, 69], [53, 57, 60, 65]]  # C G Am F
    bass = [36, 31, 33, 29]; beat = 60 / 112; bar = 4 * beat; b = 0
    while b * bar < DUR:
        ch, t0 = prog[b % 4], b * bar
        pluck(bass[b % 4], t0, .3, 2, 2); pluck(bass[b % 4] + 7, t0 + 2 * beat, .2, 2.5, 1.5)
        for k, idx in enumerate([0, 2, 1, 3, 2, 1, 3, 2]):
            pluck(ch[idx] + 12, t0 + k * beat / 2, .12 if k % 2 == 0 else .08, 5, .8)
        if b >= 2:
            for k, bt in enumerate((0, 1.5, 3)):
                pluck(ch[(b + k) % 4] + 24, t0 + bt * beat, .07, 3, 1.2)
        b += 1
    out = out[:n]; dl = int(.28 * sr); out[dl:] += .25 * out[:-dl]
    env = np.ones(n); env[:sr] = np.linspace(0, 1, sr); env[-3 * sr:] = np.linspace(1, 0, 3 * sr)
    out = out * env / np.abs(out).max() * .8
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((out * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    wav = OUT + ".wav"; music(wav)
    if len(sys.argv) > 2:  # önizleme: belirtilen saniyelerin karelerini kaydet
        for s in sys.argv[2:]:
            Image.frombytes("RGB", (W, H), render(int(float(s) * FPS))).save(f"{OUT}_{s}.jpg")
        sys.exit()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as pool:
        for b in pool.imap(render, range(int(DUR * FPS)), chunksize=8): ff.stdin.write(b)
    ff.stdin.close(); assert ff.wait() == 0, "ffmpeg hata verdi"
    os.remove(wav)

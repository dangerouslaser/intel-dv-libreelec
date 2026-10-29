"""Score candidate renders against reference captures (crop rows 540..1619, cols 960..2879).
usage: score.py <refdir> <renderdir> <pattern-group> ...   (renders named <idx>-<variant>.rgb16)"""
import sys, glob, json, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import dvtunnel as t, dvlms as d, numpy as np
Y0, Y1, X0, X1 = 540, 1620, 960, 2880
H, W = Y1 - Y0, X1 - X0
KYCC = np.array([[9574, 0, 13802], [9574, -1540, -5348], [9574, 17610, 0]], float) / 8192
KOFF = np.array([1 / 16, 0.5, 0.5])
KLMS = np.array([[7222, 8771, 390], [2654, 12430, 1300], [0, 422, 15962]], float) / 16384

def lms_from_codes(Y, C, ycc, off, lms, cb_even=True):
    Yn = Y.astype(float) / 4096; Cn = C.astype(float) / 4096
    ev, od = Cn[:, 0::2], Cn[:, 1::2]
    cb, cr = (ev, od) if cb_even else (od, ev)
    v = np.stack([Yn - off[0], np.repeat(cb, 2, 1) - off[1], np.repeat(cr, 2, 1) - off[2]])
    lin = d.pq_eotf(np.einsum("ij,jhw->ihw", ycc, v))
    return d.pq_oetf(np.einsum("ij,jhw->ihw", lms, lin)) * 4095

def pack(rgb, mode="pair"):
    """Kodi engine tunnel packing (BT.2020 limited YCbCr 4:2:2); mode selects chroma decimation."""
    r, g, b = [np.clip(x, 0, 1) for x in rgb]
    Yl = 0.2627 * r + 0.6780 * g + 0.0593 * b
    y = np.floor(256 + 3504 * Yl + 0.5).astype(np.int32)
    cbf = (b - Yl) / 1.8814; crf = (r - Yl) / 1.4746
    def dec(x):
        if mode == "pair":            # current engine: average of the two pixels
            return (x[:, 0::2] + x[:, 1::2]) / 2
        if mode == "cosited":         # take the even pixel
            return x[:, 0::2]
        if mode == "121":             # [1 2 1]/4 centred on the even pixel
            xp = np.pad(x, ((0, 0), (1, 1)), mode="edge")
            return (xp[:, 0:-2:2] + 2 * xp[:, 1:-1:2] + xp[:, 2::2]) / 4
        taps = {"14641": [1, 4, 6, 4, 1], "halfband": [-1, 0, 9, 16, 9, 0, -1],
                "lanczos2": None}[mode]
        if taps is None:              # Lanczos2 at half rate, centred on the even pixel
            import math
            def l2(t): t = abs(t); return 1.0 if t < 1e-9 else (0.0 if t >= 2 else 2 * math.sin(math.pi * t) * math.sin(math.pi * t / 2) / (math.pi * t) ** 2)
            taps = [l2(k / 2) for k in range(-3, 4)]
        taps = np.array(taps, float); taps /= taps.sum(); r = len(taps) // 2
        xp = np.pad(x, ((0, 0), (r, r)), mode="edge")
        out = sum(taps[k] * xp[:, k:k + x.shape[1]] for k in range(len(taps)))
        return out[:, 0::2]
        raise ValueError(mode)
    c = np.empty_like(y)
    c[:, 0::2] = np.floor(2048 + 3584 * dec(cbf) + 0.5)
    c[:, 1::2] = np.floor(2048 + 3584 * dec(crf) + 0.5)
    return y, c

def ref_crop(path):
    y, c = t.load_ce(path)
    ycc, off, lms = d.matrices(y, c)
    return lms_from_codes(y[Y0:Y1, X0:X1], c[Y0:Y1, X0:X1], ycc, off, lms)

def render_lms(path, mode="pair"):
    a = np.fromfile(path, dtype=np.uint16).reshape(H, W, 3).astype(float).transpose(2, 0, 1) / 65535
    y, c = pack(a, mode)
    return lms_from_codes(y, c, KYCC, KOFF, KLMS)

def edges(C):
    opp = C[1] - C[2]
    g = np.zeros_like(opp)
    g[1:-1, 1:-1] = np.abs(opp[1:-1, 2:] - opp[1:-1, :-2]) + np.abs(opp[2:, 1:-1] - opp[:-2, 1:-1])
    m = np.zeros(g.shape, bool); m[4:-4, 4:-4] = True
    return m, m & (g > np.percentile(g[m], 98))

def score(C, X, m, e):
    D = C - X
    D = D - D[:, m].mean(axis=1)[:, None, None]   # remove the constant offset
    return float(np.sqrt((D[:, m] ** 2).mean())), float(np.sqrt((D[:, e] ** 2).mean()))

# ---- IPT-PQ-c2 tunnel emulation (what Dolby hardware sends) ----
_ipt_cache = {}
def ipt_matrices(ref_path):
    if "m" not in _ipt_cache:
        y, c = t.load_ce(ref_path)
        _ipt_cache["m"] = d.matrices(y, c)
    return _ipt_cache["m"]

def ipt_codes(rgb, ycc, off, lms):
    """RGB PQ (BT.2020) -> IPT-PQ-c2 code values (float, 12-bit scale, full range)."""
    lin = d.pq_eotf(np.clip(rgb, 0, 1))
    L = np.einsum("ij,jhw->ihw", KLMS, lin)                      # Dolby LMS
    x = np.einsum("ij,jhw->ihw", np.linalg.inv(lms), L)           # undo crosstalk
    nl = d.pq_oetf(x)
    return (np.einsum("ij,jhw->ihw", np.linalg.inv(ycc), nl) + off[:, None, None]) * 4096

def render_lms_ipt(path, ref_path, mode="cosited", shift=(0.0, 0.0, 0.0)):
    ycc, off, lms = ipt_matrices(ref_path)
    a = np.fromfile(path, dtype=np.uint16).reshape(H, W, 3).astype(float).transpose(2, 0, 1) / 65535
    I, P, T = ipt_codes(a, ycc, off, lms)
    I = I + shift[0]; P = P + shift[1]; T = T + shift[2]
    def dec(x):
        if mode == "cosited": return x[:, 0::2]
        if mode == "pair": return (x[:, 0::2] + x[:, 1::2]) / 2
        xp = np.pad(x, ((0, 0), (1, 1)), mode="edge")
        return (xp[:, 0:-2:2] + 2 * xp[:, 1:-1:2] + xp[:, 2::2]) / 4
    y = np.clip(np.floor(I + 0.5), 0, 4095).astype(np.int32)
    c = np.empty_like(y)
    c[:, 0::2] = np.clip(np.floor(dec(P) + 0.5), 0, 4095)
    c[:, 1::2] = np.clip(np.floor(dec(T) + 0.5), 0, 4095)
    return lms_from_codes(y, c, ycc, off, lms)

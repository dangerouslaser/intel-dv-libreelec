"""Convert a decoded DV tunnel frame into Dolby LMS (PQ-encoded, 12-bit code units) using the
matrices carried in its own DM metadata packet, so signals in different spaces
(BT.2020 YCbCr from the Kodi engine, IPT-PQ-c2 from Amlogic) can be compared."""
import numpy as np
import dvtunnel as t

M1, M2 = 2610 / 16384, 2523 / 4096 * 128
C1, C2, C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_eotf(e):
    e = np.clip(e, 0, 1)
    p = e ** (1 / M2)
    return (np.maximum(p - C1, 0) / (C2 - C3 * p)) ** (1 / M1)


def pq_oetf(y):
    y = np.clip(y, 0, 1)
    p = y ** M1
    return ((C1 + C2 * p) / (1 + C3 * p)) ** M2


def packet(y, c, pk=0):
    yf, cf = y.reshape(-1), c.reshape(-1)
    idx = np.arange(pk * 3072, pk * 3072 + 1024)
    cc, yy = cf[idx], yf[idx]
    par = (np.array([bin(v >> 1).count("1") for v in cc]) + np.array([bin(v).count("1") for v in yy])) & 1
    return np.packbits(((cc & 1) ^ par).astype(np.uint8)).tobytes()


def matrices(y, c):
    p = packet(y, c)[5:]  # skip the 5-byte packet header; payload byte 0,1 = flags
    o = 2
    s16 = lambda i: int.from_bytes(p[i:i + 2], "big", signed=True)
    u32 = lambda i: int.from_bytes(p[i:i + 4], "big")
    ycc = np.array([s16(o + 2 * k) for k in range(9)], float).reshape(3, 3) / 8192
    o += 18
    off = np.array([u32(o + 4 * k) for k in range(3)], float) / (1 << 28)
    o += 12
    lms = np.array([s16(o + 2 * k) for k in range(9)], float).reshape(3, 3) / 16384
    return ycc, off, lms


def to_lms_pq(frame, cb_even=True):
    """Return (3, H-4, W) array: PQ-encoded LMS in 12-bit code units."""
    y, c = frame
    ycc, off, lms = matrices(y, c)
    Y = y[4:].astype(np.float64) / 4096
    C = c[4:].astype(np.float64) / 4096
    ev, od = C[:, 0::2], C[:, 1::2]
    cb, cr = (ev, od) if cb_even else (od, ev)
    Cb = np.repeat(cb, 2, axis=1)
    Cr = np.repeat(cr, 2, axis=1)
    v = np.stack([Y - off[0], Cb - off[1], Cr - off[2]])
    nl = np.einsum("ij,jhw->ihw", ycc, v)
    lin = pq_eotf(nl)
    L = np.einsum("ij,jhw->ihw", lms, lin)
    return pq_oetf(L) * 4095

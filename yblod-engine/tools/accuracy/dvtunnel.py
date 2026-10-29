"""Decode Dolby Vision tunnel frames (12-bit ICtCp 4:2:2 in 8-bit RGB) from two capture sources
and compare them.

  Kodi (Petunia) capture: RGBA8 3840x2160 from glReadPixels (row order to be detected).
  AM9 Pro (CoreELEC VDIN1) capture: RGB8 3840x2160 with every 64-bit word byte-reversed.

Tunnel pixel: R = C[11:4], G = Y[11:4], B = Y[3:0] | C[3:0] << 4; C alternates Cb/Cr.
The first rows carry DV metadata in the chroma LSB (3072 pixels per 128-byte packet).
"""
import sys
import numpy as np

W, H = 3840, 2160


def crc32_mpeg2(data):
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def unpack_rgb(r, g, b):
    y = (g.astype(np.int32) << 4) | (b & 15)
    c = (r.astype(np.int32) << 4) | (b >> 4)
    return y, c


def packets_ok(y, c, n=2):
    """Number of leading metadata packets with a valid CRC (checks row order)."""
    yf, cf = y.reshape(-1), c.reshape(-1)
    good = 0
    for pk in range(n):
        idx = np.arange(pk * 3072, pk * 3072 + 1024)
        cc, yy = cf[idx], yf[idx]
        parity = (np.array([bin(v >> 1).count("1") for v in cc]) + np.array([bin(v).count("1") for v in yy])) & 1
        bits = (cc & 1) ^ parity
        by = np.packbits(bits.astype(np.uint8)).tobytes()
        if crc32_mpeg2(by[:124]) == int.from_bytes(by[124:], "big"):
            good += 1
    return good


def load_kodi(path):
    a = np.fromfile(path, dtype=np.uint8).reshape(H, W, 4)
    for flip in (False, True):
        f = a[::-1] if flip else a
        y, c = unpack_rgb(f[..., 0], f[..., 1], f[..., 2])
        if packets_ok(y, c):
            return y, c
    raise ValueError(f"{path}: no valid DV metadata packet in either row order")


def load_ce(path):
    a = np.fromfile(path, dtype=np.uint8)
    a = a.reshape(-1, 8)[:, ::-1].reshape(H, W, 3)
    y, c = unpack_rgb(a[..., 2], a[..., 0], a[..., 1])
    if not packets_ok(y, c):
        raise ValueError(f"{path}: no valid DV metadata packet")
    return y, c


def active_rows(y):
    """Skip the metadata rows (first 2 rows at most) when comparing pictures."""
    return slice(4, H)


def diff_stats(a, b):
    ya, ca = a
    yb, cb = b
    s = active_rows(ya)
    out = {}
    for name, p, q in (("Y", ya[s], yb[s]), ("C", ca[s], cb[s])):
        d = np.abs(p - q).astype(np.float64)
        out[name] = dict(rmse=float(np.sqrt((d ** 2).mean())), mean=float(d.mean()), max=int(d.max()),
                         ge1=float((d >= 1).mean() * 100), ge4=float((d >= 4).mean() * 100),
                         ge16=float((d >= 16).mean() * 100))
    return out


def fmt(st):
    return "  ".join(f"{k}: rmse {v['rmse']:6.2f} max {v['max']:4d} >=1 {v['ge1']:5.1f}% >=4 {v['ge4']:5.2f}% >=16 {v['ge16']:5.2f}%"
                     for k, v in st.items())


def thumb(y):
    return y[4::16, ::16].astype(np.float64)


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    y, c = (load_kodi if kind == "kodi" else load_ce)(path)
    print(path, "Y", y.min(), y.max(), "C", c.min(), c.max(), "mean Y", y[4:].mean())

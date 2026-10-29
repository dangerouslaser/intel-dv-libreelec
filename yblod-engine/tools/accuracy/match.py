import sys, glob, json
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import dvtunnel as t, dvlms as d, numpy as np
LMS = np.array([[7222, 8771, 390], [2654, 12430, 1300], [0, 422, 15962]], float) / 16384
keys, th = [], []
for f in sorted(glob.glob("thumbs/thumb-*.u16"), key=lambda f: int(f.split("-")[1].split(".")[0])):
    keys.append(int(f.split("-")[1].split(".")[0]))
    a = np.fromfile(f, dtype=np.uint16).reshape(135, 240, 3).astype(float).transpose(2, 0, 1) / 65535
    th.append(d.pq_oetf(np.einsum("ij,jhw->ihw", LMS, d.pq_eotf(a))) * 4095)
stack = np.stack(th)
res = []
n = len(glob.glob("ce/ce-*.raw"))
for i in range(n):
    try:
        c = d.to_lms_pq(t.load_ce(f"ce/ce-{i}.raw"), True)
    except Exception as e:
        print(f"ce-{i}: {e}"); continue
    ct = c[:, 12::16, ::16][:, :134]
    sc = np.abs(stack[:, :, 1:135, :] - ct[None]).mean(axis=(1, 2, 3))
    o = np.argsort(sc)
    res.append((i, keys[o[0]], float(sc[o[0]]), float(sc[o[1]]), bool(sc[o[1]] > 2 * sc[o[0]])))
print("unique matches:", sum(r[4] for r in res), "of", len(res))
print("frames:", " ".join(str(r[1]) for r in res if r[4]))
json.dump(res, open("matches.json", "w"))

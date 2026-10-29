#!/usr/bin/env python3
"""Summarize qualified captures; timings are overlapping, not additive."""
import collections
import json
import math
import pathlib
import re
import statistics
import sys

def stats(values):
    values = sorted(values)
    return dict(n=len(values), median=statistics.median(values),
                p95=values[math.ceil(len(values)*.95)-1], mean=statistics.mean(values)) if values else None

result = {}
for path in sorted(pathlib.Path(sys.argv[1]).glob('*.json')):
    if path.name.startswith(('pilot-', 'summary')):
        continue
    data = json.loads(path.read_text())
    expected_window = 10000 if data['overlay'] == 'home' else 12005
    if (data.get('qualified_window') is False or
            data['window']['currentwindow']['id'] != expected_window):
        print(f'Skipping unqualified capture: {path.name}', file=sys.stderr)
        continue
    groups = collections.defaultdict(list)
    for event in data['gpu_events']:
        for kind in ('source_wait_pts', 'source_async_profile_pts', 'source_piecewise_profile_pts',
                     'source_pack_profile_pts', 'overlay_profile_pts'):
            if kind in event:
                if kind == 'source_wait_pts' and bool(event['overlay']) != (data['overlay'] == 'home'):
                    continue
                if event.get('valid', 1) == 1:
                    groups[kind].append(event)
    out = {'window': data['window'], 'transitions': data.get('transitions', []),
           'gpu_host_groups': {}}
    for kind, events in groups.items():
        events = events[24:]
        keys = {k for e in events for k in e if k.endswith('_ms')}
        out['gpu_host_groups'][kind] = {k: stats([e[k] for e in events if k in e]) for k in sorted(keys)}
        if kind == 'source_wait_pts':
            pts = collections.Counter(e[kind] for e in events)
            out['repeated_pts_after_warmup'] = sum(n-1 for n in pts.values())
    lines = [l for l in data['timing_lines'] if 'transition:' in l and
             ('overlay=true' if data['overlay'] == 'home' else 'overlay=false') in l][24:]
    out['kodi_ms'] = {}
    for key in ('prior_gl_ms', 'render_call_ms', 'drain_ms', 'processing_ms'):
        out['kodi_ms'][key] = stats([float(m.group(1)) for l in lines
            for m in [re.search(r'\b' + key + r'=([\d.]+)', l)] if m])
    out['memory_MiB_per_second'] = {}
    for line in (data.get('memory_perf') or '').splitlines():
        fields = line.split(',')
        if len(fields) >= 5 and fields[1] == 'MiB':
            out['memory_MiB_per_second'][fields[2]] = float(fields[0])/(float(fields[3])/1e9)
    out['health'] = [l for l in data['timing_lines'] if 'playback health:' in l]
    result[path.stem] = out
print(json.dumps(result, indent=2))

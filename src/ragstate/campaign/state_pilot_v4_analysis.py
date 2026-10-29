"""Auditable Track-A v4 analysis after campaign-wide hard validity checks.

The invalid v1-v3 scientific observations are never read. All tables and
figures are derived from the completed v4 canonical observation rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import binomtest, chi2

from .state_pilot_manifest import _load_frozen, digest
from .state_pilot_v4_campaign import (
    CAMPAIGN, CONFIG, PREFLIGHT, ROOT, deterministic_fingerprint,
    execute_recorded, validate_scenario, verify_freeze,
)


FIGURES = ROOT / 'figures'
AUTH = frozenset(('GRANT', 'REVOKE'))
CONTENT = frozenset(('UPDATE', 'SUPERSEDE', 'DELETE'))
FIELDS = ('scenario_id', 'query_id', 'domain', 'query_family', 'logical_time',
          'block', 'transition', 'fault', 'severity', 'architecture',
          'scrs', 'cbr', 'all_claims_success', 'srr', 'dep', 'mvcr',
          'ucer_candidate', 'ucer_post_filter', 'ucer_context', 'ardr', 'cter',
          'serr', 'invalid_evidence_absence_success', 'state_correct',
          'cache_hit', 'cache_invalidations', 'replica_refreshes',
          'auth_validation_count', 'retrieval_latency_ms', 'index_converged',
          'convergence_status', 't_rsc', 'partial_topology',
          'partial_total_units', 'partial_updated_units',
          'partial_stale_units', 'partial_realized_fraction',
          'partial_matched_uniform_scenario_id', 'chunk_index_hash',
          'parent_index_hash', 'auth_metadata_hash', 'cache_hash',
          'context_hash', 'canonical_source_state_hash',
          'canonical_authorization_state_hash')


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _table(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True)
                             if isinstance(row.get(key), (dict, list)) else row.get(key)
                             for key in fields})


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _p95(values):
    values = sorted(value for value in values if value is not None)
    return values[min(len(values)-1, math.ceil(.95*len(values))-1)] if values else None


def _row_at(rows, tick):
    return next(row for row in rows if row['logical_time'] == tick)


def _group_summary(groups, *, tick=1):
    result = []
    for key, scenario_groups in sorted(groups.items(), key=lambda item: str(item[0])):
        observed = [_row_at(rows, tick) for rows in scenario_groups]
        convergence = [rows[-1]['t_rsc'] for rows in scenario_groups]
        result.append({'group': key, 'n': len(observed), 'tick': tick,
                       'scrs': _mean(row['scrs'] for row in observed),
                       'cbr': _mean(row['cbr'] for row in observed),
                       'claim_success': _mean(row['all_claims_success'] for row in observed),
                       'state_correct': _mean(row['state_correct'] for row in observed),
                       'ucer_context': _mean(row['ucer_context'] for row in observed),
                       'ardr': _mean(row['ardr'] for row in observed),
                       'dep': _mean(row['dep'] for row in observed),
                       'srr': _mean(row['srr'] for row in observed),
                       'mvcr': _mean(row['mvcr'] for row in observed),
                       'cter': _mean(row['cter'] for row in observed),
                       'serr': _mean(row['serr'] for row in observed),
                       't_rsc_median_uncensored': median([value for value in convergence
                                                         if value is not None])
                       if any(value is not None for value in convergence) else None,
                       'censored': sum(value is None for value in convergence),
                       'latency_median_ms': median(row['retrieval_latency_ms'] for row in observed),
                       'latency_p95_ms': _p95(row['retrieval_latency_ms'] for row in observed),
                       'cache_hit_rate': _mean(row['cache_hit'] for row in observed),
                       'auth_validation_mean': _mean(row['auth_validation_count'] for row in observed)})
    return result


def _km(rows_by_scenario, horizon=14):
    durations = []
    for rows in rows_by_scenario:
        value = rows[-1]['t_rsc']
        durations.append((value if value is not None else horizon,
                          value is not None))
    at_risk = len(durations)
    survival = 1.0
    curve = [(0, 1.0)]
    rmst = 0.0
    previous = 0
    km_median = None
    for tick in sorted({time for time, event in durations if event}):
        rmst += survival * (tick - previous)
        event_count = sum(time == tick and event for time, event in durations)
        censor_count = sum(time == tick and not event for time, event in durations)
        if at_risk:
            survival *= (1 - event_count / at_risk)
        curve.append((tick, survival))
        if km_median is None and survival <= .5:
            km_median = tick
        at_risk -= event_count + censor_count
        previous = tick
    rmst += survival * (horizon - previous)
    return {'n': len(durations), 'events': sum(event for _, event in durations),
            'censored': sum(not event for _, event in durations),
            'km_median': km_median, 'rmst_to_14': rmst, 'curve': curve}


def _logrank(group_a, group_b, horizon=14):
    a = [(rows[-1]['t_rsc'] if rows[-1]['t_rsc'] is not None else horizon,
          rows[-1]['t_rsc'] is not None) for rows in group_a]
    b = [(rows[-1]['t_rsc'] if rows[-1]['t_rsc'] is not None else horizon,
          rows[-1]['t_rsc'] is not None) for rows in group_b]
    observed = expected = variance = 0.0
    for tick in sorted({time for time, event in a+b if event}):
        n1 = sum(time >= tick for time, _ in a)
        n2 = sum(time >= tick for time, _ in b)
        d1 = sum(time == tick and event for time, event in a)
        d2 = sum(time == tick and event for time, event in b)
        n, d = n1+n2, d1+d2
        if n <= 1:
            continue
        observed += d1
        expected += d*n1/n
        variance += n1*n2*d*(n-d)/(n*n*(n-1))
    statistic = ((observed-expected)**2/variance) if variance > 0 else 0.0
    return {'logrank_chi2': statistic,
            'p_value': float(chi2.sf(statistic, 1)) if variance > 0 else None,
            'group_a_n': len(a), 'group_b_n': len(b)}


def _paired(a, b, *, seed=606, replicates=2000):
    """Risk difference b-a, paired bootstrap, and exact McNemar."""
    x = np.asarray(a, dtype=int)
    y = np.asarray(b, dtype=int)
    if len(x) != len(y):
        raise ValueError('Paired comparison lengths differ')
    diff = y-x
    rng = np.random.default_rng(seed)
    boot = np.mean(diff[rng.integers(0, len(diff), size=(replicates, len(diff)))], axis=1)
    discordant_ab = int(np.sum((x == 1) & (y == 0)))
    discordant_ba = int(np.sum((x == 0) & (y == 1)))
    discordant = discordant_ab + discordant_ba
    return {'n': len(diff), 'risk_difference': float(np.mean(diff)),
            'ci95_low': float(np.quantile(boot, .025)),
            'ci95_high': float(np.quantile(boot, .975)),
            'discordant_a_only': discordant_ab,
            'discordant_b_only': discordant_ba,
            'mcnemar_exact_p': float(binomtest(min(discordant_ab, discordant_ba),
                                              discordant, .5).pvalue)
            if discordant else 1.0}


def _bh(rows):
    eligible = sorted([(index, row['mcnemar_exact_p']) for index, row in enumerate(rows)
                       if row.get('secondary') and row.get('mcnemar_exact_p') is not None],
                      key=lambda item: item[1])
    qvalues = [None]*len(rows)
    adjusted = 1.0
    for rank in range(len(eligible), 0, -1):
        index, pvalue = eligible[rank-1]
        adjusted = min(adjusted, pvalue*len(eligible)/rank)
        qvalues[index] = adjusted
    for row, value in zip(rows, qvalues):
        row['bh_fdr_q'] = value
    return rows


def _reproduction_subset(scenarios):
    specs = (
        ('UPDATE', lambda s: s['block']=='TA1' and s['transition']=='UPDATE' and s['severity']=='S2'),
        ('REVOKE', lambda s: s['block']=='TA1' and s['transition']=='REVOKE' and s['severity']=='S2'),
        ('DELETE', lambda s: s['block']=='TA1' and s['transition']=='DELETE' and s['severity']=='S2'),
        ('PARTIAL', lambda s: s.get('partial_topology')=='P_MID'),
        ('OUT_OF_ORDER', lambda s: s['fault']=='OUT_OF_ORDER'),
        ('CACHE', lambda s: s['block']=='TA4' and s['architecture']=='A0'),
        ('MITIGATION', lambda s: s['block']=='TA3' and s['architecture']=='A4' and s['transition']=='REVOKE'),
        ('COMPOUND', lambda s: s.get('compound')=='CF6' and s['component']=='F1_F2'),
    )
    return [{'stratum': label,
             'scenario_id': sorted((s['scenario_id'] for s in scenarios if match(s)))[0]}
            for label, match in specs]


def reproduce_fresh():
    config, manifest, _ = verify_freeze()
    world, metadata, queries, _ = _load_frozen(config)
    by_qid = {query.query_id: query for query in queries}
    by_sid = {scenario['scenario_id']: scenario for scenario in manifest['scenarios']}
    output = []
    for selected in _reproduction_subset(manifest['scenarios']):
        scenario = by_sid[selected['scenario_id']]
        query = by_qid[scenario['query_id']]
        rows, accesses, cache_ops, snapshots = execute_recorded(
            scenario, world, metadata, query, config)
        validity = validate_scenario(scenario, rows, accesses, cache_ops,
                                     snapshots, query, metadata, world, config)
        output.append({**selected,
                       'fingerprint': deterministic_fingerprint(rows, accesses,
                                                                cache_ops, validity)})
    return output


def check_reproduction():
    config, manifest, _ = verify_freeze()
    stored = []
    for selected in _reproduction_subset(manifest['scenarios']):
        sid = selected['scenario_id']
        rows = _read_jsonl(ROOT / 'scenario_rows' / (sid + '.jsonl'))
        accesses = _read_jsonl(ROOT / 'canonical_access_rows' / (sid + '.jsonl'))
        cache_ops = _read_jsonl(ROOT / 'cache_stage_rows' / (sid + '.jsonl'))
        validity = json.loads((ROOT / 'validity_rows' / (sid + '.json')).read_text())
        stored.append({**selected,
            'fingerprint': deterministic_fingerprint(rows, accesses, cache_ops, validity)})
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path('src').resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    fresh = subprocess.run([sys.executable, '-m',
        'ragstate.campaign.state_pilot_v4_analysis', 'reproduce'],
        capture_output=True, text=True, env=env)
    observed = json.loads(fresh.stdout) if fresh.returncode == 0 else None
    result = {'identical': observed == stored, 'fresh_process_exit': fresh.returncode,
              'subset': stored, 'fresh': observed, 'stderr': fresh.stderr}
    _json(ROOT / 'reproduction.json', result)
    return result


def hard_validity():
    config, manifest, provenance = verify_freeze()
    wiring = json.loads((PREFLIGHT / 'wiring_gate_summary.json').read_text())
    execution = json.loads((ROOT / 'execution_summary.json').read_text())
    access = json.loads((ROOT / 'canonical_access_audit.json').read_text())
    cache = json.loads((ROOT / 'cache_stage_audit.json').read_text())
    agreement = json.loads((ROOT / 'compiler_runtime_agreement.json').read_text())
    if any(wiring.get(f'W{i}') != 'PASS' for i in range(1, 7)):
        raise ValueError('Wiring preflight failed')
    if (ROOT / 'campaign_invalid.json').exists():
        raise ValueError('Campaign was marked INVALID')
    if execution['scenario_count'] != 4242 or execution['observation_rows'] != 63630:
        raise ValueError('Campaign incomplete')
    if sum(execution['status_counts'].values()) != 4242 or set(execution['status_counts']) - {'valid','censored'}:
        raise ValueError('Invalid or failed scenario in campaign')
    if execution['ta0']['D1'] != 'PASS' or access['forbidden'] or cache['violations'] or \
       agreement['agreements'] != 4242 or agreement['mismatches']:
        raise ValueError('Campaign-wide hard scientific-validity gate failed')
    return {'D1': 'PASS', 'D2a': 'PASS', 'D2b': 'PASS',
            'D2c': 'PASS', 'D2d': 'PASS'}


def load_compact():
    by_scenario = defaultdict(list)
    with (ROOT / 'observations.jsonl').open() as stream:
        for line in stream:
            original = json.loads(line)
            row = {key: original.get(key) for key in FIELDS}
            by_scenario[row['scenario_id']].append(row)
    for rows in by_scenario.values():
        rows.sort(key=lambda item: item['logical_time'])
        if len(rows) != 15:
            raise ValueError('Incomplete trajectory in consolidated rows')
    return by_scenario


class _Canvas:
    """Small dependency-free SVG/PNG plotting surface for reproducible figures."""
    WIDTH, HEIGHT = 900, 540
    COLORS = ('#2463aa', '#c95a32', '#32875b', '#8a58a9', '#d49d22', '#4b6e73')
    FONT = {
        'A':('01110','10001','10001','11111','10001','10001','10001'),
        'B':('11110','10001','10001','11110','10001','10001','11110'),
        'C':('01111','10000','10000','10000','10000','10000','01111'),
        'D':('11110','10001','10001','10001','10001','10001','11110'),
        'E':('11111','10000','10000','11110','10000','10000','11111'),
        'F':('11111','10000','10000','11110','10000','10000','10000'),
        'G':('01111','10000','10000','10111','10001','10001','01111'),
        'H':('10001','10001','10001','11111','10001','10001','10001'),
        'I':('11111','00100','00100','00100','00100','00100','11111'),
        'J':('00111','00010','00010','00010','10010','10010','01100'),
        'K':('10001','10010','10100','11000','10100','10010','10001'),
        'L':('10000','10000','10000','10000','10000','10000','11111'),
        'M':('10001','11011','10101','10101','10001','10001','10001'),
        'N':('10001','11001','10101','10011','10001','10001','10001'),
        'O':('01110','10001','10001','10001','10001','10001','01110'),
        'P':('11110','10001','10001','11110','10000','10000','10000'),
        'Q':('01110','10001','10001','10001','10101','10010','01101'),
        'R':('11110','10001','10001','11110','10100','10010','10001'),
        'S':('01111','10000','10000','01110','00001','00001','11110'),
        'T':('11111','00100','00100','00100','00100','00100','00100'),
        'U':('10001','10001','10001','10001','10001','10001','01110'),
        'V':('10001','10001','10001','10001','10001','01010','00100'),
        'W':('10001','10001','10001','10101','10101','10101','01010'),
        'X':('10001','10001','01010','00100','01010','10001','10001'),
        'Y':('10001','10001','01010','00100','00100','00100','00100'),
        'Z':('11111','00001','00010','00100','01000','10000','11111'),
        '0':('01110','10001','10011','10101','11001','10001','01110'),
        '1':('00100','01100','00100','00100','00100','00100','01110'),
        '2':('01110','10001','00001','00010','00100','01000','11111'),
        '3':('11110','00001','00001','01110','00001','00001','11110'),
        '4':('00010','00110','01010','10010','11111','00010','00010'),
        '5':('11111','10000','10000','11110','00001','00001','11110'),
        '6':('01110','10000','10000','11110','10001','10001','01110'),
        '7':('11111','00001','00010','00100','01000','01000','01000'),
        '8':('01110','10001','10001','01110','10001','10001','01110'),
        '9':('01110','10001','10001','01111','00001','00001','01110'),
        ' ':('00000',)*7, '.':('00000',)*6+('00100',),
        ':':('00000','00100','00100','00000','00100','00100','00000'),
        '-':('00000','00000','00000','11111','00000','00000','00000'),
        '/':('00001','00001','00010','00100','01000','10000','10000'),
        '_':('00000',)*6+('11111',),
        '(':('00010','00100','01000','01000','01000','00100','00010'),
        ')':('01000','00100','00010','00010','00010','00100','01000'),
        '=':('00000','11111','00000','11111','00000','00000','00000'),
        '+':('00000','00100','00100','11111','00100','00100','00000'),
    }

    def __init__(self, title):
        self.title = title
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.WIDTH}" '
                    f'height="{self.HEIGHT}" viewBox="0 0 {self.WIDTH} {self.HEIGHT}">',
                    '<rect width="100%" height="100%" fill="white"/>']
        self.pixels = bytearray([255, 255, 255] * self.WIDTH * self.HEIGHT)
        self.text(70, 38, title, 23, 'start')

    def _rgb(self, color):
        return tuple(int(color[index:index+2], 16) for index in (1, 3, 5))

    def rect(self, x, y, width, height, color):
        x, y, width, height = map(int, (x, y, width, height))
        self.svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}"/>')
        rgb = self._rgb(color)
        for yy in range(max(0, y), min(self.HEIGHT, y+height)):
            for xx in range(max(0, x), min(self.WIDTH, x+width)):
                offset = (yy*self.WIDTH + xx)*3
                self.pixels[offset:offset+3] = bytes(rgb)

    def line(self, x1, y1, x2, y2, color, thickness=2):
        x1,y1,x2,y2=map(int,(x1,y1,x2,y2))
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                        f'stroke="{color}" stroke-width="{thickness}"/>')
        steps = max(abs(x2-x1), abs(y2-y1), 1)
        for step in range(steps+1):
            xx = round(x1+(x2-x1)*step/steps)
            yy = round(y1+(y2-y1)*step/steps)
            for dx in range(-thickness//2, thickness//2+1):
                for dy in range(-thickness//2, thickness//2+1):
                    if 0 <= xx+dx < self.WIDTH and 0 <= yy+dy < self.HEIGHT:
                        offset = ((yy+dy)*self.WIDTH + xx+dx)*3
                        self.pixels[offset:offset+3] = bytes(self._rgb(color))

    def text(self, x, y, value, size=13, anchor='middle'):
        safe = (str(value).replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;'))
        self.svg.append(f'<text x="{int(x)}" y="{int(y)}" font-family="Arial" '
                        f'font-size="{size}" text-anchor="{anchor}" fill="#27384a">{safe}</text>')
        # Raster preview labels use the same deterministic, local bitmap font.
        scale = 2 if size >= 12 else 1
        label = str(value).upper()
        width = len(label)*6*scale
        left = int(x) if anchor == 'start' else int(x-width/2)
        top = int(y)-7*scale
        for index, letter in enumerate(label):
            glyph = self.FONT.get(letter, self.FONT[' '])
            for gy, pattern in enumerate(glyph):
                for gx, bit in enumerate(pattern):
                    if bit == '1':
                        px, py = left+(index*6+gx)*scale, top+gy*scale
                        for dy in range(scale):
                            for dx in range(scale):
                                if 0 <= px+dx < self.WIDTH and 0 <= py+dy < self.HEIGHT:
                                    offset = ((py+dy)*self.WIDTH+px+dx)*3
                                    self.pixels[offset:offset+3] = b'\x27\x38\x4a'

    def save(self, stem):
        FIGURES.mkdir(exist_ok=True)
        self.svg.append('</svg>')
        (FIGURES / (stem + '.svg')).write_text('\n'.join(self.svg) + '\n')
        raw = b''.join(b'\x00' + bytes(self.pixels[row*self.WIDTH*3:(row+1)*self.WIDTH*3])
                       for row in range(self.HEIGHT))
        def chunk(tag, body):
            return struct.pack('!I', len(body)) + tag + body + \
                struct.pack('!I', zlib.crc32(tag+body) & 0xffffffff)
        png = (b'\x89PNG\r\n\x1a\n' +
               chunk(b'IHDR', struct.pack('!2I5B', self.WIDTH, self.HEIGHT,
                                         8, 2, 0, 0, 0)) +
               chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))
        (FIGURES / (stem + '.png')).write_bytes(png)


def _bars(stem, title, labels, values, *, ymax=None):
    chart = _Canvas(title)
    values = [float(value or 0) for value in values]
    ymax = ymax or max(values + [1.0])
    chart.line(75, 455, 850, 455, '#293a49')
    chart.line(75, 75, 75, 455, '#293a49')
    if values:
        slot = 760/len(values)
        for index, (label, value) in enumerate(zip(labels, values)):
            x = 85 + index*slot
            height = max(1, int(340*value/ymax))
            chart.rect(x, 455-height, max(3, slot*.64), height,
                       chart.COLORS[index % len(chart.COLORS)])
            chart.text(x+slot*.32, 475, str(label)[:14], 11)
            chart.text(x+slot*.32, 445-height, f'{value:.3f}', 11)
    chart.save(stem)


def _lines(stem, title, curves, *, ymin=0, ymax=1):
    chart = _Canvas(title)
    chart.line(75, 455, 850, 455, '#293a49')
    chart.line(75, 75, 75, 455, '#293a49')
    for index, (label, points) in enumerate(curves.items()):
        color = chart.COLORS[index % len(chart.COLORS)]
        ordered = sorted(points)
        coords = [(80 + 760*x/14, 455 - 360*(y-ymin)/(ymax-ymin))
                  for x,y in ordered]
        for a,b in zip(coords, coords[1:]):
            chart.line(*a, *b, color, 3)
        chart.rect(590, 72+index*21, 16, 12, color)
        chart.text(612, 83+index*21, label, 12, 'start')
    for tick in (0,2,5,10,14):
        chart.text(80+760*tick/14, 475, tick, 12)
    chart.save(stem)


def _scenario_groups(by_scenario, manifest, *, block=None, transition=None,
                     architecture=None, severity=None, fault=None):
    for scenario in manifest['scenarios']:
        if block is not None and scenario['block'] != block: continue
        if transition is not None and scenario['transition'] != transition: continue
        if architecture is not None and scenario['architecture'] != architecture: continue
        if severity is not None and scenario['severity'] != severity: continue
        if fault is not None and scenario['fault'] != fault: continue
        yield scenario, by_scenario[scenario['scenario_id']]


def build_tables(by_scenario, manifest):
    scenarios = manifest['scenarios']
    composition = Counter((scenario['block'], scenario['transition'],
                           scenario['fault'], scenario['severity'],
                           scenario['architecture']) for scenario in scenarios)
    _table(ROOT / 'scenario_composition.csv', [
        {'block': key[0], 'transition': key[1], 'fault': key[2],
         'severity': key[3], 'architecture': key[4], 'scenario_count': count}
        for key, count in sorted(composition.items())])
    # Isolated transition trajectories, retaining every domain and severity.
    transition_groups = defaultdict(list)
    for scenario, rows in _scenario_groups(by_scenario, manifest, block='TA1', architecture='A0'):
        transition_groups[scenario['transition'], scenario['severity'], rows[0]['domain']].append(rows)
    transition_rows = []
    for (transition, severity, domain), items in sorted(transition_groups.items()):
        for tick in (1, 5, 10, 14):
            row = _group_summary({(transition,severity,domain): items}, tick=tick)[0]
            row.update(transition=transition, severity=severity, domain=domain)
            row.pop('group')
            transition_rows.append(row)
    _table(ROOT / 'transition_summary.csv', transition_rows)

    convergence_groups = defaultdict(list)
    for scenario in scenarios:
        convergence_groups[(scenario['block'],scenario['transition'],
                            scenario['severity'],scenario['architecture'])].append(
                            by_scenario[scenario['scenario_id']])
    convergence_rows = []
    for (block,transition,severity,architecture), items in sorted(convergence_groups.items()):
        km = _km(items)
        convergence_rows.append({'block': block, 'transition': transition,
            'severity': severity, 'architecture': architecture,
            'n': km['n'], 'events': km['events'], 'censored': km['censored'],
            'km_median': km['km_median'], 'rmst_to_14': km['rmst_to_14'],
            'km_curve': km['curve']})
    _table(ROOT / 'convergence_summary.csv', convergence_rows)

    authorization_rows = []
    for block in ('TA1','TA3'):
        groups = defaultdict(list)
        for scenario, rows in _scenario_groups(by_scenario, manifest, block=block):
            if scenario['transition'] in ('GRANT','REVOKE'):
                groups[scenario['transition'],scenario['architecture'],scenario['severity']].append(rows)
        for (transition,architecture,severity), items in sorted(groups.items()):
            for tick in (1,5,10):
                observed = [_row_at(rows,tick) for rows in items]
                exposure_windows = [sum(row['ucer_context'] > 0
                                        for row in rows if row['logical_time'] >= 0)
                                    for rows in items] if transition == 'REVOKE' else []
                authorization_rows.append({'block': block, 'transition': transition,
                    'architecture': architecture, 'severity': severity, 'tick': tick,
                    'n': len(items),
                    'ucer_candidate': _mean(row['ucer_candidate'] for row in observed),
                    'ucer_post_filter': _mean(row['ucer_post_filter'] for row in observed),
                    'ucer_context': _mean(row['ucer_context'] for row in observed),
                    'ardr': _mean(row['ardr'] for row in observed),
                    'rew_observed_ticks_median': median(exposure_windows)
                    if exposure_windows else None,
                    'auth_validation_mean': _mean(row['auth_validation_count'] for row in observed)})
    _table(ROOT / 'authorization_summary.csv', authorization_rows)

    fault_groups = defaultdict(list)
    for scenario, rows in _scenario_groups(by_scenario, manifest, block='TA2'):
        fault_groups[(scenario['transition'],scenario['fault'],
                      scenario.get('partial_topology') or 'none')].append(rows)
    fault_rows = []
    for (transition,fault,topology), items in sorted(fault_groups.items()):
        at5 = [_row_at(rows,5) for rows in items]
        at6 = [_row_at(rows,6) for rows in items]
        fault_rows.append({'transition':transition,'fault':fault,'topology':topology,
            'n':len(items),'scrs_at_5':_mean(r['scrs'] for r in at5),
            'claim_success_at_5':_mean(r['all_claims_success'] for r in at5),
            'state_correct_at_5':_mean(r['state_correct'] for r in at5),
            'state_correct_at_6':_mean(r['state_correct'] for r in at6),
            'ucer_at_5':_mean(r['ucer_context'] for r in at5),
            'dep_at_5':_mean(r['dep'] for r in at5),
            'srr_at_5':_mean(r['srr'] for r in at5),
            'serr_any':_mean(any(r['serr'] for r in rows) for rows in items),
            'censored':sum(rows[-1]['t_rsc'] is None for rows in items)})
    _table(ROOT / 'fault_mode_summary.csv', fault_rows)

    partial_pairs = []
    for scenario in scenarios:
        if not scenario.get('partial_topology'):
            continue
        partial = by_scenario[scenario['scenario_id']]
        uniform = by_scenario[scenario['matched_uniform_scenario_id']]
        p5,u5 = _row_at(partial,5),_row_at(uniform,5)
        p6,u6 = _row_at(partial,6),_row_at(uniform,6)
        partial_pairs.append({'partial_scenario_id':scenario['scenario_id'],
            'uniform_scenario_id':scenario['matched_uniform_scenario_id'],
            'query_id':scenario['query_id'], 'domain':p5['domain'],
            'transition':scenario['transition'],'architecture':scenario['architecture'],
            'severity':scenario['severity'],
            'partial_topology':scenario['partial_topology'],
            'unit_count':scenario['total_unit_count'],
            'updated_units':scenario['updated_unit_count'],
            'stale_units':scenario['total_unit_count']-scenario['updated_unit_count'],
            'realized_fraction':scenario['realized_fraction'],
            'selected_updated_ids_hash':scenario['selected_updated_ids_hash'],
            'uniform_index_converged_at_5':u5['index_converged'],
            'partial_index_converged_at_5':p5['index_converged'],
            'uniform_scrs_at_5':u5['scrs'],'partial_scrs_at_5':p5['scrs'],
            'uniform_claim_success_at_5':u5['all_claims_success'],
            'partial_claim_success_at_5':p5['all_claims_success'],
            'uniform_failure_at_5':1-u5['state_correct'],
            'partial_failure_at_5':1-p5['state_correct'],
            'uniform_failure_at_6':1-u6['state_correct'],
            'partial_failure_at_6':1-p6['state_correct'],
            'uniform_t_rsc':uniform[-1]['t_rsc'],
            'partial_t_rsc':partial[-1]['t_rsc'],
            'comparison_interpretation': 'mixed_partial_vs_fully_propagated_at_t5'})
    _table(ROOT / 'partial_vs_uniform.csv', partial_pairs)
    fractions = defaultdict(list)
    for row in partial_pairs:
        fractions[(row['transition'],row['partial_topology'],row['realized_fraction'])].append(row)
    fraction_rows = []
    for (transition,topology,fraction), items in sorted(fractions.items()):
        fraction_rows.append({'transition':transition,'partial_topology':topology,
            'realized_fraction':fraction,'n':len(items),
            'partial_failure_at_5':_mean(item['partial_failure_at_5'] for item in items),
            'uniform_failure_at_5':_mean(item['uniform_failure_at_5'] for item in items),
            'paired_difference_at_5':_mean(item['partial_failure_at_5']-
                                           item['uniform_failure_at_5'] for item in items),
            'partial_t_rsc_median_uncensored':median([item['partial_t_rsc'] for item in items
                                                    if item['partial_t_rsc'] is not None])
            if any(item['partial_t_rsc'] is not None for item in items) else None})
    _table(ROOT / 'partial_fraction_analysis.csv', fraction_rows)

    cache_rows = []
    for scenario, rows in _scenario_groups(by_scenario, manifest, block='TA4'):
        persistent = [r['logical_time'] for r in rows if r['logical_time'] >= 0
                      and r['index_converged'] and r['cache_hit'] and not r['state_correct']]
        cache_rows.append({'scenario_id':scenario['scenario_id'],
            'query_id':scenario['query_id'],'domain':rows[0]['domain'],
            'transition':scenario['transition'],'architecture':scenario['architecture'],
            'cache_persistence_after_index_convergence':bool(persistent),
            'persistence_ticks':persistent,
            'cache_hits_post_transition':sum(r['cache_hit'] for r in rows
                                             if r['logical_time'] >= 0),
            'cache_invalidations':rows[-1]['cache_invalidations'],
            't_rsc':rows[-1]['t_rsc']})
    _table(ROOT / 'cache_persistence.csv', cache_rows)

    mitigation_groups = defaultdict(list)
    for scenario,rows in _scenario_groups(by_scenario,manifest,block='TA3'):
        mitigation_groups[scenario['transition'],scenario['severity'],scenario['architecture']].append(rows)
    mitigation_rows=[]
    for (transition,severity,architecture),items in sorted(mitigation_groups.items()):
        at1=[_row_at(rows,1) for rows in items]
        at5=[_row_at(rows,5) for rows in items]
        km=_km(items)
        mitigation_rows.append({'transition':transition,'severity':severity,
            'architecture':architecture,'n':len(items),
            'failure_at_1':_mean(1-r['state_correct'] for r in at1),
            'failure_at_5':_mean(1-r['state_correct'] for r in at5),
            'ucer_at_1':_mean(r['ucer_context'] for r in at1),
            'ardr_at_1':_mean(r['ardr'] for r in at1),
            'dep_at_1':_mean(r['dep'] for r in at1),
            'km_median':km['km_median'],'rmst_to_14':km['rmst_to_14'],
            'censored':km['censored'],
            'median_latency_ms':median(r['retrieval_latency_ms'] for r in at1),
            'mean_auth_validation_calls':_mean(r['auth_validation_count'] for r in at1),
            'mean_cache_invalidations':_mean(rows[-1]['cache_invalidations'] for rows in items)})
    _table(ROOT / 'mitigation_summary.csv',mitigation_rows)
    _table(ROOT / 'architecture_overhead.csv', [
        {'transition': row['transition'], 'severity': row['severity'],
         'architecture': row['architecture'], 'n': row['n'],
         'median_latency_ms': row['median_latency_ms'],
         'mean_auth_validation_calls': row['mean_auth_validation_calls'],
         'mean_cache_invalidations': row['mean_cache_invalidations']}
        for row in mitigation_rows])

    factorial_groups=defaultdict(list)
    for scenario,rows in _scenario_groups(by_scenario,manifest,block='TA6'):
        factorial_groups[scenario['component'],tuple(scenario['component_tuple'])].append(rows)
    factorial_rows=[]
    for (block,components),items in sorted(factorial_groups.items()):
        at1=[_row_at(rows,1) for rows in items]
        at5=[_row_at(rows,5) for rows in items]
        factorial_rows.append({'factorial_block':block,'E':components[0],
            'C':components[1],'A':components[2],'n':len(items),
            'failure_at_1':_mean(1-r['state_correct'] for r in at1),
            'failure_at_5':_mean(1-r['state_correct'] for r in at5),
            'ucer_at_1':_mean(r['ucer_context'] for r in at1),
            'dep_at_5':_mean(r['dep'] for r in at5),
            'cache_hit_at_5':_mean(r['cache_hit'] for r in at5),
            'median_latency_ms':median(r['retrieval_latency_ms'] for r in at1),
            'mean_auth_validations':_mean(r['auth_validation_count'] for r in at1)})
    _table(ROOT / 'component_factorial.csv',factorial_rows)

    compound_groups=defaultdict(list)
    for scenario,rows in _scenario_groups(by_scenario,manifest,block='TA5'):
        compound_groups[scenario['compound'],scenario['architecture'],scenario['component']].append(rows)
    compound_rows=[]
    for (compound,architecture,arm),items in sorted(compound_groups.items()):
        at1=[_row_at(rows,1) for rows in items]
        at5=[_row_at(rows,5) for rows in items]
        compound_rows.append({'compound':compound,'architecture':architecture,
            'arm':arm,'n':len(items),
            'failure_at_1':_mean(1-r['state_correct'] for r in at1),
            'failure_at_5':_mean(1-r['state_correct'] for r in at5),
            'ucer_at_1':_mean(r['ucer_context'] for r in at1),
            'dep_at_5':_mean(r['dep'] for r in at5),
            't_rsc_median_uncensored':median([rows[-1]['t_rsc'] for rows in items
                                            if rows[-1]['t_rsc'] is not None])
            if any(rows[-1]['t_rsc'] is not None for rows in items) else None,
            'censored':sum(rows[-1]['t_rsc'] is None for rows in items)})
    _table(ROOT / 'compound_summary.csv',compound_rows)

    localization=Counter()
    for scenario in scenarios:
        rows=by_scenario[scenario['scenario_id']]
        pre=rows[0]
        for row in rows[1:]:
            if row['state_correct']:
                continue
            plane='unattributed_context_or_clean_retrieval'
            if row['cache_hit'] and row['index_converged']:
                plane='candidate_cache_after_index_convergence'
            elif scenario['transition'] in AUTH and row['auth_metadata_hash']==pre['auth_metadata_hash']:
                plane='stale_acl_metadata'
            elif scenario['transition'] in CONTENT and row['chunk_index_hash']==pre['chunk_index_hash']:
                plane='stale_chunk_index'
            elif scenario['transition'] in CONTENT and row['parent_index_hash']==pre['parent_index_hash']:
                plane='stale_parent_index'
            elif row['ucer_context']:
                plane='context_authorization_exposure'
            localization[(scenario['block'],plane)] += 1
    localization_rows=[{'block':block,'inferred_plane':plane,'incorrect_observation_count':count,
                        'method':'state-hash and boundary heuristic; not causal attribution'}
                       for (block,plane),count in sorted(localization.items())]
    _table(ROOT / 'localization_summary.csv',localization_rows)
    return {'transition':transition_rows,'convergence':convergence_rows,
            'authorization':authorization_rows,'fault':fault_rows,
            'partial_pairs':partial_pairs,'partial_fractions':fraction_rows,
            'cache':cache_rows,'mitigation':mitigation_rows,
            'factorial':factorial_rows,'compound':compound_rows,
            'localization':localization_rows}


def statistical_comparisons(by_scenario, manifest, tables):
    scenarios=manifest['scenarios']
    rows=[]
    for transition in ('UPDATE','SUPERSEDE','DELETE','GRANT','REVOKE'):
        items=[trajectory for scenario,trajectory in _scenario_groups(
            by_scenario,manifest,block='TA1',transition=transition,
            architecture='A0',severity='S2')]
        pre=[1-_row_at(trajectory,-1)['state_correct'] for trajectory in items]
        early=[1-_row_at(trajectory,1)['state_correct'] for trajectory in items]
        rows.append({'comparison':'TA1_clean_vs_early_'+transition,
                     'endpoint':'state_failure','secondary':False,
                     **_paired(pre,early)})
    for transition in ('SUPERSEDE','DELETE','GRANT','REVOKE'):
        group=defaultdict(dict)
        for scenario,trajectory in _scenario_groups(by_scenario,manifest,
                                                     block='TA3',transition=transition,
                                                     severity='S2'):
            group[scenario['query_id']][scenario['architecture']]=trajectory
        for architecture in ('A1','A2','A3','A4','A5'):
            matched=[pair for pair in group.values()
                     if 'A0' in pair and architecture in pair]
            for endpoint in ('state_failure','ucer_context','ardr'):
                def outcome(trajectory):
                    observed=_row_at(trajectory,1)
                    return 1-observed['state_correct'] if endpoint=='state_failure' else \
                           int(observed[endpoint]>0)
                rows.append({'comparison':f'TA3_{transition}_{architecture}_vs_A0',
                             'endpoint':endpoint,'secondary':endpoint!='state_failure',
                             **_paired([outcome(pair['A0']) for pair in matched],
                                       [outcome(pair[architecture]) for pair in matched])})
    partial=tables['partial_pairs']
    for transition in ('UPDATE','SUPERSEDE','DELETE','REVOKE'):
        items=[item for item in partial if item['transition']==transition]
        rows.append({'comparison':'PARTIAL_vs_uniform_at_delivery_'+transition,
                     'endpoint':'state_failure_at_t5_not_pure_H4',
                     'secondary':True,
                     **_paired([item['uniform_failure_at_5'] for item in items],
                               [item['partial_failure_at_5'] for item in items])})
    _table(ROOT / 'statistical_comparisons.csv',_bh(rows))
    logrank=[]
    for transition in ('SUPERSEDE','DELETE','GRANT','REVOKE'):
        a0=[trajectory for scenario,trajectory in _scenario_groups(by_scenario,manifest,
            block='TA3',transition=transition,severity='S2',architecture='A0')]
        for architecture in ('A2','A5'):
            alt=[trajectory for scenario,trajectory in _scenario_groups(by_scenario,manifest,
                block='TA3',transition=transition,severity='S2',architecture=architecture)]
            logrank.append({'transition':transition,'comparison':f'{architecture}_vs_A0',
                            **_logrank(a0,alt)})
    _table(ROOT / 'survival_comparisons.csv',logrank)
    interactions=[]
    for compound in ('CF1','CF2','CF3','CF4','CF5','CF6'):
        for architecture in ('A0','A5'):
            by_query=defaultdict(dict)
            for scenario,trajectory in _scenario_groups(by_scenario,manifest,
                block='TA5',architecture=architecture):
                if scenario['compound']==compound:
                    by_query[scenario['query_id']][scenario['component']]=trajectory
            complete=[item for item in by_query.values()
                      if all(arm in item for arm in ('NONE','F1','F2','F1_F2'))]
            for tick in (1,5):
                contrasts=[]
                for item in complete:
                    failure={arm:1-_row_at(item[arm],tick)['state_correct']
                             for arm in ('NONE','F1','F2','F1_F2')}
                    contrasts.append(failure['F1_F2']-failure['F1']-
                                     failure['F2']+failure['NONE'])
                if contrasts:
                    rng=np.random.default_rng(606)
                    sample=np.asarray(contrasts)
                    bootstrap=np.mean(sample[rng.integers(0,len(sample),
                        size=(2000,len(sample)))],axis=1)
                    interactions.append({'compound':compound,'architecture':architecture,
                        'tick':tick,'n_complete_matched_queries':len(contrasts),
                        'paired_interaction_contrast':float(np.mean(sample)),
                        'ci95_low':float(np.quantile(bootstrap,.025)),
                        'ci95_high':float(np.quantile(bootstrap,.975)),
                        'interpretation':'pilot paired descriptive contrast; no logistic claim'})
    _table(ROOT / 'compound_interactions.csv',interactions)
    return {'paired':rows,'logrank':logrank,'compound_interactions':interactions}


def make_figures(by_scenario, manifest, tables):
    scenarios=manifest['scenarios']
    transitions=('UPDATE','SUPERSEDE','DELETE','GRANT','REVOKE')
    curves={}
    for transition in transitions:
        items=[trajectory for scenario,trajectory in _scenario_groups(by_scenario,manifest,
            block='TA1',transition=transition,severity='S2',architecture='A0')]
        curves[transition]=[(tick,_mean(_row_at(rows,tick)['scrs'] for rows in items))
                            for tick in (0,1,2,3,4,5,6,7,8,9,10,11,12,14)]
    _lines('01_transition_trajectories','SCRS trajectories: A0, S2',curves)

    survival={}
    for severity in ('S1','S2','S3'):
        items=[trajectory for scenario,trajectory in _scenario_groups(by_scenario,manifest,
            block='TA1',architecture='A0',severity=severity)]
        curve=_km(items)['curve']
        survival[severity]=[(tick,1-value) for tick,value in curve]
    _lines('02_convergence_survival','Stable convergence: cumulative probability',survival)

    for transition,metric,stem,title in (
        ('REVOKE','ucer_context','03_revoke_ucer','REVOKE context UCER by architecture'),
        ('GRANT','ardr','04_grant_ardr','GRANT availability denial by architecture')):
        traces={}
        for architecture in ('A0','A1','A2','A3','A4','A5'):
            items=[trajectory for scenario,trajectory in _scenario_groups(by_scenario,manifest,
                block='TA3',transition=transition,severity='S2',architecture=architecture)]
            traces[architecture]=[(tick,_mean(_row_at(rows,tick)[metric] for rows in items))
                                  for tick in (0,1,2,3,4,5,6,7,8,9,10,11,12,14)]
        _lines(stem,title,traces)

    persistence=Counter(item['transition'] for item in tables['cache']
                        if item['cache_persistence_after_index_convergence'])
    _bars('05_cache_persistence','Cache persistence after index convergence',
          list(persistence) or ['none'],list(persistence.values()) or [0])
    partial=tables['partial_pairs']
    _bars('06_partial_vs_uniform','Mixed partial vs matched delay at t=5 (control current)',
          list(PARTIAL_TRANSITIONS := ('UPDATE','SUPERSEDE','DELETE','REVOKE')),
          [_mean(item['partial_failure_at_5']-item['uniform_failure_at_5']
                 for item in partial if item['transition']==transition)
           for transition in PARTIAL_TRANSITIONS],ymax=1)
    fractions=sorted({item['realized_fraction'] for item in partial})
    _bars('07_realized_fraction','Partial failure at delivery by realized fraction',
          [f'{fraction:.2f}' for fraction in fractions],
          [_mean(item['partial_failure_at_5'] for item in partial
                 if item['realized_fraction']==fraction) for fraction in fractions],ymax=1)
    mitigations=tables['mitigation']
    _bars('08_mitigation','A0-A5 early failure: S2 focused transitions',
          ['A0','A1','A2','A3','A4','A5'],
          [_mean(row['failure_at_1'] for row in mitigations
                 if row['architecture']==architecture and row['severity']=='S2')
           for architecture in ('A0','A1','A2','A3','A4','A5')],ymax=1)
    factorial=tables['factorial']
    _bars('09_component_factorial','Component-factorial early failure',
          [f"{row['factorial_block'][:4]}:{row['E']}{row['C']}{row['A']}"
           for row in factorial],
          [row['failure_at_1'] for row in factorial],ymax=1)
    compound=tables['compound']
    _bars('10_compounds','Compound arms: A0 failure at t=5',
          [f"{row['compound']}:{row['arm']}" for row in compound
           if row['architecture']=='A0'],
          [row['failure_at_5'] for row in compound
           if row['architecture']=='A0'],ymax=1)
    planes=Counter()
    for row in tables['localization']:
        planes[row['inferred_plane']]+=row['incorrect_observation_count']
    _bars('11_localization','Incorrect observations by inferred state plane',
          list(planes),list(planes.values()))


def _rew_rows(by_scenario, manifest):
    data=[]
    for scenario in manifest['scenarios']:
        if scenario['transition']!='REVOKE' or scenario['block'] not in ('TA1','TA3'):
            continue
        rows=[row for row in by_scenario[scenario['scenario_id']]
              if row['logical_time']>=0]
        exposed=[row['logical_time'] for row in rows if row['ucer_context']>0]
        if not exposed:
            duration=0
            censored=False
        else:
            stable=[row['logical_time'] for row in rows
                    if row['logical_time']>max(exposed)]
            duration=stable[0] if stable else 14
            censored=not bool(stable)
        data.append({'scenario_id':scenario['scenario_id'],'block':scenario['block'],
                     'architecture':scenario['architecture'],'severity':scenario['severity'],
                     'domain':rows[0]['domain'],'exposed_ticks':len(exposed),
                     'rew_end_tick':duration,'right_censored':censored})
    _table(ROOT/'rew_context.csv',data)
    return data


def assess_gates(by_scenario,manifest,tables,reproduction,hard):
    scenarios=manifest['scenarios']
    # The frozen scientific tests are evaluated transparently after hard validity.
    effects={}
    specificity={}
    for transition in ('UPDATE','SUPERSEDE','DELETE','GRANT','REVOKE'):
        items=[rows for scenario,rows in _scenario_groups(by_scenario,manifest,
            block='TA1',transition=transition,architecture='A0')
            if scenario['severity'] in ('S2','S3')]
        pre=_mean(1-_row_at(rows,-1)['state_correct'] for rows in items)
        early=_mean(1-_row_at(rows,1)['state_correct'] for rows in items)
        at1=[_row_at(rows,1) for rows in items]
        effects[transition]={'n':len(items),'pre_failure':pre,
                             'early_failure':early,'risk_increase':early-pre}
        specificity[transition]={metric:_mean(row[metric] for row in at1)
                                 for metric in ('dep','ucer_context','srr','ardr','scrs')}
    observable=sum(item['risk_increase']>0 for item in effects.values())
    d3=observable>=4
    d4=(specificity['DELETE']['dep']>0 and specificity['DELETE']['ucer_context']==0
        and specificity['REVOKE']['ucer_context']>0 and specificity['REVOKE']['dep']==0
        and specificity['SUPERSEDE']['srr']>0
        and specificity['GRANT']['ardr']>0
        and specificity['UPDATE']['scrs']<1)
    ta1={}
    for severity in ('S1','S2','S3'):
        values=[rows[-1]['t_rsc'] for scenario,rows in _scenario_groups(by_scenario,manifest,
            block='TA1',architecture='A0',severity=severity)
            if rows[-1]['t_rsc'] is not None]
        ta1[severity]=median(values) if values else None
    d5=len({value for value in ta1.values() if value is not None})>=2
    mitigations=tables['mitigation']
    def mitigation(transition,arch,metric):
        vals=[row[metric] for row in mitigations if row['transition']==transition
              and row['architecture']==arch and row['severity']=='S2']
        return vals[0] if vals else None
    revoke_a0=mitigation('REVOKE','A0','ucer_at_1')
    revoke_a4=mitigation('REVOKE','A4','ucer_at_1')
    periodic=mitigation('SUPERSEDE','A1','km_median')
    event=mitigation('SUPERSEDE','A2','km_median')
    cache_a0=sum(row['cache_persistence_after_index_convergence'] for row in tables['cache']
                 if row['architecture']=='A0')
    cache_a3=sum(row['cache_persistence_after_index_convergence'] for row in tables['cache']
                 if row['architecture']=='A3')
    targeted={'live_auth_revoke':revoke_a0 is not None and revoke_a4 is not None
              and revoke_a4<revoke_a0,
              'event_vs_periodic_supersede':periodic is not None and event is not None
              and event<periodic,
              'epoch_cache_persistence':cache_a3<cache_a0}
    d6=sum(targeted.values())>=2
    d7=any(row['cache_persistence_after_index_convergence'] for row in tables['cache'])
    compound_distinct=[]
    for compound in ('CF1','CF2','CF3','CF4','CF5','CF6'):
        by_query=defaultdict(dict)
        for scenario,rows in _scenario_groups(by_scenario,manifest,block='TA5',architecture='A0'):
            if scenario['compound']==compound:
                by_query[scenario['query_id']][scenario['component']]=rows
        count=0
        for arms in by_query.values():
            if all(arm in arms for arm in ('F1','F2','F1_F2')):
                compound_trace=tuple(row['state_correct'] for row in arms['F1_F2'])
                f1=tuple(row['state_correct'] for row in arms['F1'])
                f2=tuple(row['state_correct'] for row in arms['F2'])
                if compound_trace!=f1 and compound_trace!=f2:
                    count+=1
        compound_distinct.append({'compound':compound,'matched_query_distinctions':count})
    d8=any(item['matched_query_distinctions']>0 for item in compound_distinct)
    domain_effects=defaultdict(set)
    for scenario,rows in _scenario_groups(by_scenario,manifest,block='TA1',
                                          architecture='A0',severity='S2'):
        if _row_at(rows,1)['state_correct']<_row_at(rows,-1)['state_correct']:
            domain_effects[scenario['transition']].add(rows[0]['domain'])
    d9=sum(len(domains)>=2 for domains in domain_effects.values())>=4
    factorial=tables['factorial']
    factor_effects=[]
    for block in ('authorization_revoke_S2','content_cache_S2'):
        items=[row for row in factorial if row['factorial_block']==block]
        for factor in ('E','C','A'):
            on=[row['failure_at_1'] for row in items if row[factor]==1]
            off=[row['failure_at_1'] for row in items if row[factor]==0]
            if on and off:
                factor_effects.append({'block':block,'factor':factor,
                                       'risk_difference':_mean(on)-_mean(off)})
    d10=any(abs(item['risk_difference'])>0 for item in factor_effects)
    # H4's frozen matched trajectories do not provide a same-tick uniform-stale
    # control while the partial state is mixed; this is reported as a design
    # anomaly and is never interpreted as a pure heterogeneity effect.
    timing_mismatch=sum(row['uniform_index_converged_at_5']==1
                        and row['partial_index_converged_at_5']==0
                        for row in tables['partial_pairs'])
    gates={**hard,'D3':'PASS' if d3 else 'FAIL','D4':'PASS' if d4 else 'FAIL',
           'D5':'PASS' if d5 else 'FAIL','D6':'PASS' if d6 else 'FAIL',
           'D7':'PASS' if d7 else 'FAIL','D8':'PASS' if d8 else 'FAIL',
           'D9':'PASS' if d9 else 'FAIL','D10':'PASS' if d10 else 'FAIL',
           'D11':'DEFERRED_TRACK_B','D12':'PASS' if reproduction['identical'] else 'FAIL',
           'support':{'transition_effects':effects,'metric_specificity':specificity,
                      'observable_transition_count':observable,
                      'ta1_convergence_medians':ta1,
                      'targeted_mitigations':targeted,
                      'cache_persistence_A0':cache_a0,
                      'cache_persistence_A3':cache_a3,
                      'compound_distinctions':compound_distinct,
                      'cross_domain_effect_counts':{key:len(value)
                          for key,value in domain_effects.items()},
                      'factorial_main_effects':factor_effects,
                      'partial_uniform_timing_mismatch_pairs':timing_mismatch,
                      'partial_pairs_total':len(tables['partial_pairs']),
                      'h4_primary_estimable':timing_mismatch==0}}
    _json(ROOT/'gate_summary.json',gates)
    _json(ROOT/'metric_specificity.json',specificity)
    return gates


def summarize_science(tables,stats,gates,rew):
    support=gates['support']
    expected=[]
    if support['observable_transition_count']>=4:
        expected.append('At least four primary transition families perturb early state correctness under A0.')
    if gates['D4']=='PASS':
        expected.append('DELETE, REVOKE, SUPERSEDE, GRANT, and UPDATE activate their intended metric dimensions without systematic DELETE/UCER contamination.')
    if gates['D5']=='PASS':
        expected.append('Stable convergence duration varies with the predeclared delay severity.')
    nonobvious=[]
    if gates['D7']=='PASS':
        nonobvious.append('Candidate-cache-mediated incorrectness can remain after the relevant index reports convergence; the cache-stage filter still runs.')
    if gates['D8']=='PASS':
        nonobvious.append('At least one compound arm has a distinct correctness trajectory from both isolated components; interaction inference remains exploratory.')
    if support['targeted_mitigations'].get('epoch_cache_persistence'):
        nonobvious.append('Epoch-aware cache behavior changes persistence after index convergence relative to naive reuse.')
    null=[]
    for key,value in support['targeted_mitigations'].items():
        if not value:null.append(f'No targeted directional effect detected for {key} at the prespecified comparison.')
    unchanged=[item['compound'] for item in support['compound_distinctions']
               if item['matched_query_distinctions']==0]
    if unchanged:
        null.append('No compound trajectory distinct from both isolated arms for '
                    + ', '.join(unchanged) + ' in the matched A0 pilot cases.')
    for effect in support['factorial_main_effects']:
        if effect['risk_difference']==0:
            null.append(f"No early failure difference for {effect['factor']} in "
                        f"the {effect['block']} factorial block.")
    anomalies=[]
    mismatch=support['partial_uniform_timing_mismatch_pairs']
    if mismatch:
        anomalies.append(f'{mismatch}/{support["partial_pairs_total"]} PARTIAL pairs are mixed at t=5 while the matched uniform-delay control is already fully propagated. The frozen primary H4 same-time mixed-vs-uniformly-stale contrast is not identifiable from these trajectories.')
    summary=(
        '# Track-A v4 scientific pilot\n\n'
        'All 4,242 frozen scenarios completed with 63,630 observations. '
        'The hard validity gates D1/D2a/D2b/D2c/D2d passed. Track B was not run.\n\n'
        '## Gates\n\n' + '\n'.join(f'- {key}: {gates[key]}' for key in
        ('D1','D2a','D2b','D2c','D2d','D3','D4','D5','D6','D7','D8','D9','D10','D12'))+
        '\n\n## Expected validation\n\n'+ '\n'.join('- '+item for item in expected)+
        '\n\n## Potentially non-obvious candidates\n\n'+
        ('\n'.join('- '+item for item in nonobvious) if nonobvious else '- None established in this pilot.')+
        '\n\n## Null findings\n\n'+
        ('\n'.join('- '+item for item in null) if null else '- No prespecified targeted null at the reported comparison.')+
        '\n\n## Anomalies and limits\n\n'+
        ('\n'.join('- '+item for item in anomalies) if anomalies else '- No additional timing anomaly found.')+
        '\n- Controlled synthetic corpus scale is limited; no enterprise-scale claim is made.'
        '\n- Retrieval is state-isolated; PA2 ecological transfer remains untested.'
        '\n- The compound/factorial cells are small and do not support manuscript-level interaction claims.'
        '\n- Logical-tick latency is measured in a local CPU process and is not a deployment latency estimate.\n')
    (ROOT/'summary.md').write_text(summary)
    return {'expected':expected,'potentially_nonobvious':nonobvious,
            'null':null,'anomalies':anomalies}


def analyze():
    hard=hard_validity()
    reproduction=check_reproduction()
    if not reproduction['identical']:
        raise ValueError('Fresh-process reproduction failed; scientific interpretation stopped')
    _,manifest,_=verify_freeze()
    by_scenario=load_compact()
    tables=build_tables(by_scenario,manifest)
    rew=_rew_rows(by_scenario,manifest)
    stats=statistical_comparisons(by_scenario,manifest,tables)
    gates=assess_gates(by_scenario,manifest,tables,reproduction,hard)
    make_figures(by_scenario,manifest,tables)
    findings=summarize_science(tables,stats,gates,rew)
    return {'gates':{key:value for key,value in gates.items() if key!='support'},
            'support':gates['support'],'findings':findings,
            'figure_count':len(list(FIGURES.glob('*.svg'))),
            'reproduction_identical':reproduction['identical']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('analyze','reproduce'))
    args=parser.parse_args()
    print(json.dumps(reproduce_fresh() if args.action=='reproduce' else analyze(),
                     indent=2,sort_keys=True))


if __name__=='__main__':
    main()

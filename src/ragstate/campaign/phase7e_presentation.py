"""Repair Phase 7E presentation figures from frozen CSV results only.

This script never recomputes rankings, metrics, or selection. It records the
original figure hashes and writes a separate presentation amendment.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .phase7a_figures import Canvas


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regenerate(output):
    output = Path(output)
    figures = output / 'figures'
    transitions = list(csv.DictReader((output / 'reranker_transitions.csv').open()))
    paired = list(csv.DictReader((output / 'statistical_comparisons.csv').open()))
    names = ('candidate_rank_shift', 'rescue_regression', 'paired_improvement')
    original = {f'{name}.{ext}': _sha(figures / f'{name}.{ext}')
                for name in names for ext in ('svg', 'png')}

    # Required claim ranks are taken from the frozen top-30 candidate list.
    canvas = Canvas('Required claim rank before and after reranking',
                    'candidate rank (1 to 30)', 'final top-10 rank (top is best)')
    colors = {'RR1': '#c05621', 'RR2': '#2f855a'}
    canvas.legend([(name, color) for name, color in colors.items()])
    for row in transitions:
        before = json.loads(row['rr0_candidate_claim_ranks'])
        for configuration, color in colors.items():
            after = json.loads(row[f'{configuration.lower()}_final_claim_ranks'])
            for document_id, candidate_rank in before.items():
                final_rank = after.get(document_id)
                if candidate_rank is None or final_rank is None:
                    continue
                x = 85 + int((candidate_rank - 1) * 650 / 29)
                y = 70 + int((final_rank - 1) * 340 / 9)
                canvas.rect(x - 2, y - 2, x + 3, y + 3, color)
    for rank in (1, 5, 10, 20, 30):
        canvas.label(85 + int((rank - 1) * 650 / 29), 450, rank, size=10)
    canvas.save(figures / 'candidate_rank_shift')

    # Plot both rescue and regression for both models; counts are over the
    # same 204 answerable queries, so percentages are directly comparable.
    answerable = [row for row in transitions if row['query_family'] != 'negative']
    canvas = Canvas('Claim rescues and regressions', 'reranker', 'fraction of answerable queries')
    canvas.legend([('rescued', '#2f855a'), ('regressed', '#c05621')])
    for index, configuration in enumerate(('RR1', 'RR2')):
        counts = Counter(row[f'{configuration}_claim_outcome'] for row in answerable)
        for offset, outcome, color in ((0, 'rescued', '#2f855a'),
                                       (1, 'regressed', '#c05621')):
            fraction = counts[outcome] / len(answerable)
            x = 190 + index * 330 + offset * 44
            canvas.rect(x, canvas.y(fraction), x + 38, 430, color)
            canvas.label(x + 19, canvas.y(fraction) - 8, counts[outcome], size=11)
        canvas.label(210 + index * 330, 450, configuration)
    canvas.save(figures / 'rescue_regression')

    # A centered zero line retains negative paired effects instead of clipping
    # them to zero. Axis values are difference + 0.5, as labeled.
    canvas = Canvas('Paired risk differences versus RR0', 'reranker / endpoint',
                    'risk difference + 0.5 (zero at center)')
    canvas.line(75, canvas.y(0.5), 755, canvas.y(0.5), '#222222', 2)
    scopes = [('all_answerable', 'scrs', 'SCRS'),
              ('all_answerable', 'all_required_claim_success', 'claim'),
              ('multi_document', 'all_required_claim_success', 'multi')]
    for model_index, configuration in enumerate(('RR1', 'RR2')):
        for endpoint_index, (scope, endpoint, label) in enumerate(scopes):
            row = next(item for item in paired if item['configuration'] == configuration
                       and item['scope'] == scope and item['endpoint'] == endpoint)
            difference = float(row['paired_difference'])
            top = canvas.y(0.5 + difference)
            center = canvas.y(0.5)
            x = 112 + model_index * 320 + endpoint_index * 80
            canvas.rect(x, min(top, center), x + 38, max(top, center) + 1,
                        '#2f855a' if difference >= 0 else '#c05621')
            canvas.label(x + 19, min(top, center) - 8, f'{difference:+.3f}', size=10)
            canvas.label(x + 19, 450, f'{configuration} {label}', size=9)
    canvas.save(figures / 'paired_improvement')

    amendment = {'scope': 'presentation figures only; no rankings, metrics, or gates changed',
                 'source': str(Path(__file__)), 'source_sha256': _sha(__file__),
                 'replaced_figure_sha256': original,
                 'corrected_figure_sha256': {f'{name}.{ext}': _sha(figures / f'{name}.{ext}')
                     for name in names for ext in ('svg', 'png')}}
    (output / 'presentation_amendment.json').write_text(json.dumps(amendment, indent=2,
                                                               sort_keys=True) + '\n')
    return amendment


if __name__ == '__main__':
    import sys
    regenerate(sys.argv[1] if len(sys.argv) > 1 else 'results/phase7e-reranker-v1')

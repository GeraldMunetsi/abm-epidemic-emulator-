import json
import numpy as np
from pathlib import Path

BASE_DIR = Path("experiments/Ablation studies/out/results/testing")

CONDITIONS = [
    ('full', 'Full model'),
    ('no_rff', 'No RFF (raw params)'),
    ('no_spline', 'No B-spline (MLP only)'),
]

NOTES = {
    'full' : 'Baseline',
    'no_rff' : 'Skip RFF, feed params_norm directly',
    'no_spline': 'no B-spline',
}

TEST_STRATEGIES = [
    ('mcmc_test',   'MCMC test set'),
    ('lhs_test',    'LHS test set'),
    ('random_test', 'Random test set'),
]


def load_stats(condition: str, strategy_dir: str) -> dict | None:
    result_file = BASE_DIR / condition / strategy_dir / 'test_final_statistics.json'
    if not result_file.exists():
        return None
    with open(result_file) as f:
        return json.load(f)


def print_table(strategy_dir: str, strategy_label: str):
    rows = []
    for condition, label in CONDITIONS:
        stats = load_stats(condition, strategy_dir)
        if stats is None:
            rows.append({
                'label': label, 'r2_i': float('nan'), 'r2_s': float('nan'),
                'r2_r': float('nan'), 'rel_mae_i': float('nan'),
                'rel_ci_lower': float('nan'), 'rel_ci_upper': float('nan'),
                'note': '(not yet tested)',
            })
            continue

        rel = stats.get('relative_MAE_I_%', {})
        rel_ci = rel.get('ci_95', [float('nan'), float('nan')])
        rows.append({
            'label': label,
            'r2_i': stats.get('R2_I', {}).get('mean', float('nan')),
            'r2_s': stats.get('R2_S', {}).get('mean', float('nan')),
            'r2_r': stats.get('R2_R', {}).get('mean', float('nan')),
            'rel_mae_i': rel.get('mean', float('nan')),
            'rel_ci_lower': rel_ci[0],
            'rel_ci_upper': rel_ci[1],
            'note' : NOTES.get(condition, ''),
        })

    w_label = max(len(r['label']) for r in rows) + 2
    header = (f"{'Condition':<{w_label}} | {'R^2_I':>7} | {'R^2_S':>7} | {'R^2_R':>9} | "
              f"{'Rel-MAE_I':>10} | {'95% CI':>19} | Notes")
    sep = '-' * len(header)
    print()
    print(f"TEST RESULTS — {strategy_label}")
    print(sep)
    print(header)
    print(sep)

    baseline_mae = next(
        (r['rel_mae_i'] for r in rows if r['label'] == 'Full model'), float('nan')
    )
    for r in rows:
        r2_i, r2_s, r2_r = r['r2_i'], r['r2_s'], r['r2_r']
        rel, lo, hi = r['rel_mae_i'], r['rel_ci_lower'], r['rel_ci_upper']

        r2_i_str = f"{r2_i:.4f}" 
        r2_s_str = f"{r2_s:.4f}" 
        r2_r_str = f"{r2_r:.4f}" 
        rel_str  = f"{rel:.2f}%" 
        ci_str   = (f"[{lo:.2f}%, {hi:.2f}%]"
                    if not (np.isnan(lo) or np.isnan(hi)) else '  N/A  ')
        delta    = (f"  ({rel - baseline_mae:+.2f}pp)"
                    if not np.isnan(rel) and r['label'] != 'Full model' else '')
        print(f"{r['label']:<{w_label}} | {r2_i_str:>7} | {r2_s_str:>7} | {r2_r_str:>9} | "
              f"{rel_str:>10}{delta:<12} | {ci_str:>19} | {r['note']}")

    print(sep)


def main():
    for strategy_dir, strategy_label in TEST_STRATEGIES:
        print_table(strategy_dir, strategy_label)
    print()


if __name__ == '__main__':
    main()

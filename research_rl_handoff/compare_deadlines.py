"""Compare existing 4/5-week experiments without retraining."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def load_run(folder):
    p = ROOT / folder
    rows = list(csv.DictReader((p / 'evaluation.csv').open(encoding='utf-8-sig')))
    rows = [r for r in rows if r['policy'] == 'rl']
    example = json.loads((p / 'example.json').read_text(encoding='utf-8'))
    return rows, example

def main():
    four, a = load_run('results')
    five, b = load_run('results_5weeks')
    if a['total_work'] != b['total_work']:
        raise ValueError('Workload differs')
    keys = lambda rows: {(r['seed'], r['eval_seed']) for r in rows}
    if keys(four) != keys(five):
        raise ValueError('Training/evaluation seed pairs differ')
    report = {'work_units': a['total_work'], 'conditions': []}
    for label, rows, example in [('4 weeks', four, a), ('5 weeks', five, b)]:
        successful = [r for r in rows if int(r['completed'])]
        per_seed = []
        for seed in sorted({r['seed'] for r in rows}):
            subset = [r for r in successful if r['seed'] == seed]
            per_seed.append({'seed': int(seed), 'mean_cost': mean(float(r['total_cost']) for r in subset) if subset else None})
        costs = [r['mean_cost'] for r in per_seed if r['mean_cost'] is not None]
        report['conditions'].append(dict(label=label, deadline_days=example['deadline_steps'] / 3,
            evaluations=len(rows), completion_rate=mean(int(r['completed']) for r in rows),
            on_time_rate=mean(int(r['on_time']) for r in rows),
            mean_cost=mean(float(r['total_cost']) for r in successful) if successful else None,
            mean_completion_days=mean(float(r['steps']) / 3 for r in successful) if successful else None,
            components={k: mean(float(r[k]) for r in successful) if successful else None
                        for k in ['fatigue', 'anxiety', 'low_mood', 'low_energy']},
            per_seed=per_seed, seed_mean_cost_std=stdev(costs) if len(costs) > 1 else None))
    old, new = report['conditions']
    report['cost_reduction_percent'] = 100 * (old['mean_cost'] - new['mean_cost']) / old['mean_cost']
    report['paired_seed_cost_changes'] = [dict(seed=x['seed'], cost_change=y['mean_cost'] - x['mean_cost'])
                                         for x, y in zip(old['per_seed'], new['per_seed'])]
    out = ROOT / 'results_5weeks'
    (out / 'deadline_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ['4 weeks', '5 weeks']
    axes[0].bar(labels, [c['mean_cost'] for c in report['conditions']])
    axes[0].set(title='Whole-episode cost', ylabel='Dimensionless proxy')
    axes[1].bar(labels, [c['mean_completion_days'] for c in report['conditions']])
    axes[1].set(title='Mean time to complete', ylabel='Days (3 slots/day)')
    bottom = [0, 0]
    for k in ['fatigue', 'anxiety', 'low_mood', 'low_energy']:
        vals = [c['components'][k] for c in report['conditions']]
        axes[2].bar(labels, vals, bottom=bottom, label=k)
        bottom = [x + y for x, y in zip(bottom, vals)]
    axes[2].set(title='Cost components')
    axes[2].legend(fontsize=8)
    fig.suptitle('Q-learning: identical 48-unit workload')
    fig.tight_layout()
    fig.savefig(out / 'deadline_comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

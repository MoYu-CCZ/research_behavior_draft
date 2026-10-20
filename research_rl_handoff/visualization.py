from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_plots(result, total_work, deadline_weeks, outdir='results'):
    out = Path(outdir)
    out.mkdir(exist_ok=True, parents=True)
    logs = result['logs']
    days = [r['day'] for r in logs]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    for key in ['energy', 'mood', 'mentor', 'anxiety']:
        axes[0].plot(days, [r[key] for r in logs], label=key)
    axes[0].legend()
    axes[0].set_ylabel('Simulated state')
    axes[1].plot(days, [total_work - r['work_remaining'] for r in logs], label='Completed')
    axes[1].plot(days, [total_work * min(d / (7 * deadline_weeks), 1) for d in days], label='Target')
    axes[1].legend()
    axes[1].set_ylabel('Work units')
    costs = [sum(r[k] for k in ['fatigue', 'anxiety', 'low_mood', 'low_energy']) for r in result['slot_logs']]
    axes[2].plot([r['step'] / 3 for r in result['slot_logs']], costs, label='Cost per slot')
    axes[2].set(xlabel='Day', ylabel='Dimensionless proxy cost')
    for ax in axes:
        ax.axvline(deadline_weeks * 7, color='red', linestyle='--', label='Deadline')
    fig.suptitle('Research behavior planning')
    fig.tight_layout()
    fig.savefig(out / 'daily_metrics.png', dpi=160)
    plt.close(fig)
    mapping = {'实验室': 0, '宿舍': 1, '健身房': 2}
    grid = [[mapping.get(a, -1) for a in (r['actions'] + [''] * 3)[:3]] for r in logs]
    fig, ax = plt.subplots(figsize=(8, max(4, len(logs) / 8)))
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_under('lightgray')
    im = ax.imshow(grid, aspect='auto', vmin=0, vmax=2, cmap=cmap)
    ax.set(xticks=[0, 1, 2], xticklabels=['Morning', 'Afternoon', 'Evening'],
           yticks=list(range(0, len(logs), 7)), yticklabels=days[::7],
           ylabel='Day', title='Actions (gray = unexecuted)')
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    fig.tight_layout()
    fig.savefig(out / 'action_schedule.png', dpi=160)
    plt.close(fig)


def save_analysis(summary, curves, outdir='results'):
    out = Path(outdir)
    policies = list(summary)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(policies, [100 * summary[p]['completion_rate'] for p in policies])
    axes[0].set(title='Full completion rate', ylabel='Percent', ylim=(0, 105))
    axes[1].bar(policies, [100 * summary[p]['on_time_rate'] for p in policies])
    axes[1].set(title='On-time rate', ylabel='Percent', ylim=(0, 105))
    axes[2].bar(policies, [summary[p]['completed_mean_cost'] or 0 for p in policies])
    axes[2].set(title='Cost among completed runs', ylabel='Dimensionless proxy')
    fig.tight_layout()
    fig.savefig(out / 'policy_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for seed in sorted({r['seed'] for r in curves}):
        rows = [r for r in curves if r['seed'] == seed]
        for ax, key in zip(axes, ['completion_rate', 'on_time_rate', 'cost']):
            ax.plot([r['episodes'] for r in rows], [r[key] for r in rows], label=str(seed))
            ax.set(xlabel='Training episodes', ylabel=key)
            ax.legend(title='Training seed')
    fig.tight_layout()
    fig.savefig(out / 'learning_curve.png', dpi=160)
    plt.close(fig)

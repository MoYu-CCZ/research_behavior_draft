import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from agent import QLearningAgent
from config import ACTIONS
from environment import simulate_one_episode, train_agent
from visualization import save_plots, save_analysis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weeks', type=int, default=4)
    parser.add_argument('--state-mode', choices=['compact','detailed'], default='detailed')
    parser.add_argument('--model-profile', choices=['legacy','time_aware'], default='time_aware')
    parser.add_argument('--grace-days', type=int, default=7)
    parser.add_argument('--delay-mood-penalty', type=float, default=2)
    parser.add_argument('--reward-mode', choices=['legacy', 'cost'], default='cost')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--work', type=int, default=None, help='Fixed work units; default 12 * weeks')
    parser.add_argument('--episodes', type=int, default=30000)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--outdir', default='results')
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error('episode counts must be positive')
    total_work = args.weeks * 12 if args.work is None else args.work
    if args.weeks < 1 or total_work < 1:
        parser.error('weeks and work must be positive')
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'settings.json').write_text(json.dumps(vars(args) | {'total_work': total_work}, indent=2), encoding='utf-8')
    env_options = dict(grace_days=args.grace_days, delay_mood_penalty=args.delay_mood_penalty, reward_mode=args.reward_mode, model_profile=args.model_profile, state_mode=args.state_mode)
    rows, curves = [], []
    example = None
    for seed in args.seeds:
        agent = QLearningAgent(ACTIONS, gamma=args.gamma, seed=seed)
        trained = 0
        while trained < args.episodes:
            count = min(1000, args.episodes - trained)
            train_agent(agent, count, seed=seed * 100000 + trained, weeks=args.weeks, total_work=total_work, **env_options)
            trained += count
            probe = [simulate_one_episode(agent, seed=900000 + i, weeks=args.weeks, total_work=total_work, **env_options)
                     for i in range(min(20, args.eval_episodes))]
            curves.append(dict(seed=seed, episodes=trained,
                               completion_rate=mean(r['success'] for r in probe),
                               on_time_rate=mean(r['on_time'] for r in probe),
                               cost=mean(r['total_cost'] for r in probe)))
        for policy in ['rl', 'random', 'fixed', 'lab']:
            for i in range(args.eval_episodes):
                r = simulate_one_episode(agent, seed=2000000 + i, weeks=args.weeks, policy=policy, total_work=total_work, **env_options)
                rows.append(dict(seed=seed, policy=policy, eval_seed=2000000 + i,
                                 completed=int(r['success']), on_time=int(r['on_time']),
                                 completion=r['completion'], steps=r['steps'],
                                 overtime_steps=r['overtime_steps'], total_cost=r['total_cost'],
                                 reason=r['reason'], **r['costs']))
                if policy == 'rl' and example is None:
                    example = r
    with (out / 'evaluation.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'learning_curve.json').write_text(json.dumps(curves, indent=2), encoding='utf-8')
    summary = {}
    for policy in ['rl', 'random', 'fixed', 'lab']:
        subset = [r for r in rows if r['policy'] == policy]
        completed = [r for r in subset if r['completed']]
        summary[policy] = dict(completion_rate=mean(r['completed'] for r in subset),
                               on_time_rate=mean(r['on_time'] for r in subset),
                               mean_completion=mean(r['completion'] for r in subset),
                               completed_mean_cost=mean(r['total_cost'] for r in completed) if completed else None,
                               completed_mean_overtime=mean(r['overtime_steps'] for r in completed) if completed else None,
                               failed_runs=len(subset) - len(completed))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'example.json').write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    save_plots(example, total_work, args.weeks, outdir=out)
    save_analysis(summary, curves, outdir=out)

if __name__ == '__main__':
    main()







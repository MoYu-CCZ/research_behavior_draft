"""Search fixed routines on development seeds, then compare RL on held-out seeds."""
import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean
from agent import QLearningAgent
from environment import ResearchEnv, simulate_one_episode, train_agent
from config import ACTIONS
from visualization import save_plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def routine(pattern, seed, penalty=2):
    env = ResearchEnv(4, seed, total_work=48, grace_days=7,
                      delay_mood_penalty=penalty, reward_mode='cost')
    env.reset()
    while not env.done:
        a = pattern[env.t % len(pattern)]
        if a not in env.allowed_actions():
            a = 1
        env.step(a)
    return env.result()

def metrics(results):
    completed = [r for r in results if r['success']]
    return dict(completion_rate=mean(r['success'] for r in results),
                on_time_rate=mean(r['on_time'] for r in results),
                mean_cost=mean(r['total_cost'] for r in completed) if completed else None,
                mean_days=mean(r['steps']/3 for r in completed) if completed else None,
                mean_delay_days=mean(r['delay_days'] for r in completed) if completed else None,
                components={k:mean(r['costs'][k] for r in completed) if completed else None
                            for k in ['fatigue','anxiety','low_mood','low_energy']},
                failed=len(results)-len(completed))

def ranking(results):
    m = metrics(results)
    return (-m['completion_rate'], m['mean_cost'] if m['mean_cost'] is not None else 99999)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=20000)
    parser.add_argument('--eval-episodes',type=int,default=100)
    args = parser.parse_args()
    out = ROOT / 'results_grace'
    out.mkdir(exist_ok=True)
    screened = []
    count = 0
    for length in (3,6,9):
        for pattern in itertools.product(range(3), repeat=length):
            # Reject patterns that cannot nominally supply 48 lab slots in 35 days.
            if sum(pattern[i % length] == 0 for i in range(105)) < 48:
                continue
            r = routine(pattern, 31000)
            count += 1
            if r['success']:
                screened.append((r['total_cost'], pattern))
    screened.sort()
    shortlist = screened[:50]
    validated = []
    dev_seeds = list(range(32000,32020))
    for _, pattern in shortlist:
        rr = [routine(pattern,s) for s in dev_seeds]
        validated.append((ranking(rr), pattern, metrics(rr)))
    validated.sort()
    selected = validated[0][1]
    (out/'fixed_search.json').write_text(json.dumps(dict(screened=count,
        screening_seed=31000,development_seeds=dev_seeds,
        selected_pattern=selected,selected_actions=[ACTIONS[a] for a in selected],
        development_metrics=validated[0][2]),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Fixed routine selected:', [ACTIONS[a] for a in selected], flush=True)
    eval_seeds = list(range(2000000,2000000+args.eval_episodes))
    summary, all_rows = {}, []
    def record(label, rr, train_seed=None, penalty=2):
        for seed,r in zip(eval_seeds,rr):
            all_rows.append(dict(policy=label,training_seed=train_seed,eval_seed=seed,
                delay_penalty=penalty,completed=int(r['success']),on_time=int(r['on_time']),
                completion=r['completion'],steps=r['steps'],delay_days=r['delay_days'],
                total_cost=r['total_cost'],**r['costs']))
    for label, pattern in [('original_fixed',(0,0,1)),('optimized_fixed',selected),
                           ('lab_gym_dorm',(0,2,1))]:
        rr = [routine(pattern,s) for s in eval_seeds]
        summary[label] = metrics(rr)
        record(label,rr)
        if label == 'optimized_fixed':
            (out/'fixed_example.json').write_text(json.dumps(rr[0],ensure_ascii=False,indent=2),encoding='utf-8')
            save_plots(rr[0],48,4,out/'fixed_plots')
    for penalty in (1,4):
        rr = [routine(selected,s,penalty) for s in eval_seeds]
        summary['fixed_penalty_'+str(penalty)] = metrics(rr)
        record('optimized_fixed',rr,penalty=penalty)
    variants = [('legacy_gamma098','legacy',0.98),('legacy_gamma1','legacy',1.0),
                ('cost_gamma1','cost',1.0)]
    curves = []
    for label,mode,gamma in variants:
        combined, seed_metrics = [], []
        for seed in (42,43,44):
            agent = QLearningAgent(ACTIONS,gamma=gamma,seed=seed)
            for start in range(0,args.episodes,1000):
                train_agent(agent,min(1000,args.episodes-start),seed=seed*100000+start,
                            weeks=4,total_work=48,grace_days=7,delay_mood_penalty=2,reward_mode=mode)
                probe = [simulate_one_episode(agent,seed=s,weeks=4,total_work=48,
                    grace_days=7,delay_mood_penalty=2,reward_mode=mode) for s in range(900000,900010)]
                curves.append(dict(variant=label,training_seed=seed,episodes=start+min(1000,args.episodes-start),**metrics(probe)))
            rr = [simulate_one_episode(agent,seed=s,weeks=4,total_work=48,
                  grace_days=7,delay_mood_penalty=2,reward_mode=mode) for s in eval_seeds]
            record(label,rr,seed)
            combined.extend(rr)
            seed_metrics.append(dict(seed=seed,**metrics(rr)))
            print(label,seed,metrics(rr),flush=True)
            if label == 'cost_gamma1':
                for penalty in (1,4):
                    sensitivity = [simulate_one_episode(agent,seed=s,weeks=4,total_work=48,
                        grace_days=7,delay_mood_penalty=penalty,reward_mode=mode) for s in eval_seeds]
                    record(label,sensitivity,seed,penalty)
                (out/('q_table_'+str(seed)+'.json')).write_text(json.dumps(dict(
                    gamma=gamma,alpha=agent.alpha,seed=seed,
                    q_table=[dict(state=list(k),values=v) for k,v in agent.q_table.items()])),encoding='utf-8')
                if seed == 42:
                    (out/'rl_example.json').write_text(json.dumps(rr[0],ensure_ascii=False,indent=2),encoding='utf-8')
                    save_plots(rr[0],48,4,out/'rl_plots')
        summary[label] = dict(**metrics(combined),per_training_seed=seed_metrics)
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'evaluation.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (out/'learning_curve.json').write_text(json.dumps(curves,indent=2),encoding='utf-8')
    (out/'settings.json').write_text(json.dumps(dict(target_days=28,max_days=35,work=48,
        delay_mood_penalty=2,episodes=args.episodes,eval_episodes=args.eval_episodes,
        train_seeds=[42,43,44],evaluation_seeds=eval_seeds,
        fixed_pattern=[ACTIONS[a] for a in selected]),ensure_ascii=False,indent=2),encoding='utf-8')
    labels=['original_fixed','optimized_fixed','legacy_gamma098','legacy_gamma1','cost_gamma1']
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,key in zip(axes,['mean_cost','on_time_rate','mean_days']):
        ax.bar(range(len(labels)),[summary[p][key] or 0 for p in labels])
        ax.set(xticks=list(range(len(labels))),xticklabels=labels,ylabel=key)
        ax.tick_params(axis='x',rotation=35)
    fig.suptitle('48 work units: target 28 days, hard limit 35 days')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()

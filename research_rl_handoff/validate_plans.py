"""Final untouched-seed validation of frozen fixed and learned plans."""
import csv,json
from pathlib import Path
from statistics import mean,stdev
from agent import QLearningAgent
from config import ACTIONS
from environment import ResearchEnv,simulate_one_episode
from search_plans import routine,metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
out=ROOT/'results_grace'
plan=json.loads((out/'fixed_ontime_plan.json').read_text(encoding='utf-8'))['plan']
frontier=json.loads((out/'fixed_frontier.json').read_text(encoding='utf-8'))
period=json.loads((out/'fixed_search.json').read_text(encoding='utf-8'))['selected_pattern']
agents=[]
for seed in (42,43,44):
    data=json.loads((out/('warm_q_table_'+str(seed)+'.json')).read_text(encoding='utf-8'))
    a=QLearningAgent(ACTIONS,gamma=data['gamma'],alpha=data['alpha'],seed=seed)
    a.q_table={tuple(r['state']):r['values'] for r in data['q_table']}
    agents.append((seed,a))

def fixed_plan(seed,penalty):
    e=ResearchEnv(4,seed,total_work=48,grace_days=7,delay_mood_penalty=penalty)
    e.reset()
    while not e.done:
        a=plan[e.t]
        if a not in e.allowed_actions():a=1
        e.step(a)
    return e.result()

rows=[];summary={}
labels=['original_fixed','optimized_ontime','fixed_one_day','fixed_four_days','warm_rl']
for penalty in (1,2,4):
    summary[str(penalty)]={}
    for label in labels:
        rr=[]
        group=agents if label=='warm_rl' else [(None,None)]
        for training_seed,agent in group:
            for seed in range(3000000,3000100):
                if label=='warm_rl':
                    r=simulate_one_episode(agent,seed=seed,weeks=4,total_work=48,
                        grace_days=7,delay_mood_penalty=penalty,reward_mode='cost',state_mode='detailed')
                elif label=='optimized_ontime':r=fixed_plan(seed,penalty)
                else:
                    pattern=(0,0,1) if label=='original_fixed' else frontier[1]['pattern'] if label=='fixed_one_day' else period
                    r=routine(pattern,seed,penalty)
                rr.append(r)
                rows.append(dict(policy=label,penalty=penalty,training_seed=training_seed,eval_seed=seed,
                    completed=int(r['success']),on_time=int(r['on_time']),steps=r['steps'],delay_days=r['delay_days'],
                    total_cost=r['total_cost'],final_mentor=r['final_mentor'],final_mood=r['final_mood'],**r['costs']))
        summary[str(penalty)][label]=dict(**metrics(rr),mean_final_mentor=mean(r['final_mentor'] for r in rr),
            mean_final_mood=mean(r['final_mood'] for r in rr),cost_std=stdev(r['total_cost'] for r in rr))
(out/'final_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (out/'final_evaluation.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (out/'recommended_schedule.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['day','morning','afternoon','evening'])
    for day in range(28):w.writerow([day+1]+[ACTIONS[a] for a in plan[day*3:day*3+3]])
fig,axes=plt.subplots(1,3,figsize=(14,5))
for ax,key in zip(axes,['mean_cost','mean_days','mean_final_mentor']):
    ax.bar(range(len(labels)),[summary['2'][p][key] for p in labels])
    ax.set(xticks=list(range(len(labels))),xticklabels=labels,ylabel=key)
    ax.tick_params(axis='x',rotation=35)
fig.suptitle('Frozen plans on new seeds: target 28 days, maximum 35 days')
fig.tight_layout();fig.savefig(out/'final_comparison.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))

"""Offline Q-learning initialization from a development-selected fixed plan,
followed by online Q-learning; inference uses learned Q values only."""
import csv,json
from pathlib import Path
from statistics import mean
from agent import QLearningAgent
from config import ACTIONS
from environment import ResearchEnv,simulate_one_episode,train_agent
from search_plans import metrics
ROOT=Path(__file__).resolve().parent
out=ROOT/'results_grace'
plan=json.loads((out/'fixed_ontime_plan.json').read_text(encoding='utf-8'))['plan']
options=dict(grace_days=7,delay_mood_penalty=2,reward_mode='cost',state_mode='detailed')
summary=[];rows=[]
for seed in (42,43,44):
    agent=QLearningAgent(ACTIONS,alpha=0.15,gamma=1,epsilon=0.1,seed=seed)
    for ep in range(2000):
        env=ResearchEnv(4,seed*100000+ep,total_work=48,**options);obs=env.reset()
        while not env.done:
            a=plan[env.t]
            if a not in env.allowed_actions():a=1
            nxt,reward,terminated,truncated,_=env.step(a)
            agent.learn(obs,a,reward,nxt,terminated or truncated,env.allowed_actions());obs=nxt
    train_agent(agent,10000,decay_interval=500,decay_rate=0.8,min_epsilon=0.005,
                seed=seed*100000+2000,weeks=4,total_work=48,**options)
    rr=[simulate_one_episode(agent,seed=s,weeks=4,total_work=48,**options)
        for s in range(2000000,2000100)]
    summary.append(dict(seed=seed,**metrics(rr)))
    for test_seed,r in zip(range(2000000,2000100),rr):
        rows.append(dict(training_seed=seed,eval_seed=test_seed,completed=int(r['success']),
                         on_time=int(r['on_time']),steps=r['steps'],delay_days=r['delay_days'],
                         total_cost=r['total_cost'],**r['costs']))
    (out/('warm_q_table_'+str(seed)+'.json')).write_text(json.dumps(dict(
        gamma=1,alpha=agent.alpha,seed=seed,state_mode='detailed',
        q_table=[dict(state=list(k),values=v) for k,v in agent.q_table.items()])),encoding='utf-8')
    print(summary[-1],flush=True)
with (out/'warm_evaluation.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(per_seed=summary,completion_rate=mean(r['completed'] for r in rows),
            on_time_rate=mean(r['on_time'] for r in rows),mean_cost=mean(r['total_cost'] for r in rows),
            mean_days=mean(r['steps']/3 for r in rows),demo_episodes=2000,online_episodes=10000,
            note='Detailed state and fixed-plan demonstration data are additional information; not a gamma-only ablation.')
(out/'warm_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)

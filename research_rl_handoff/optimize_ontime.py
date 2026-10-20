"""Allow a final-week adjustment to the best repeating fixed routine."""
import json
from pathlib import Path
from statistics import mean
from environment import ResearchEnv
from search_plans import metrics
from config import ACTIONS
ROOT=Path(__file__).resolve().parent
out=ROOT/'results_grace'
frontier=json.loads((out/'fixed_frontier.json').read_text(encoding='utf-8'))
base=frontier[1]['pattern']
schedule=[base[i%len(base)] for i in range(105)]

def run(plan,seed,penalty=2):
    env=ResearchEnv(4,seed,total_work=48,grace_days=7,delay_mood_penalty=penalty,reward_mode='cost')
    env.reset()
    while not env.done:
        a=plan[env.t]
        if a not in env.allowed_actions():a=1
        env.step(a)
    return env.result()

def score(plan):
    rr=[run(plan,s) for s in range(32000,32020)]
    return (-mean(r['success'] and r['on_time'] for r in rr),mean(r['total_cost'] for r in rr))

best=None
for i in range(84):
    if schedule[i]==0:continue
    plan=schedule.copy();plan[i]=0
    candidate=(score(plan),i,plan)
    if best is None or candidate[:2]<best[:2]:best=candidate
plan=best[2]
# Adjust recovery types without changing lab workload; develop on separate seeds.
for _ in range(2):
    changed=False
    for i in range(84):
        if plan[i]==0:continue
        candidate=plan.copy();candidate[i]=3-plan[i]
        if score(candidate)<score(plan):plan=candidate;changed=True
    if not changed:break
rr=[run(plan,s) for s in range(2000000,2000100)]
record=dict(plan=plan,actions_by_day=[[ACTIONS[a] for a in plan[d*3:d*3+3]] for d in range(28)],
            development_score=score(plan),metrics=metrics(rr),
            mean_final_mentor=mean(r['final_mentor'] for r in rr),
            mean_final_mood=mean(r['final_mood'] for r in rr))
(out/'fixed_ontime_plan.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'fixed_ontime_example.json').write_text(json.dumps(rr[0],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ('plan','actions_by_day')},indent=2))

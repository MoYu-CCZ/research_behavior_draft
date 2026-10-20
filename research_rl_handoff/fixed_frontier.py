"""Find a fixed-routine cost/time frontier on development seeds."""
import itertools,json,csv
from pathlib import Path
from statistics import mean
from search_plans import routine,metrics
from config import ACTIONS
ROOT=Path(__file__).resolve().parent
screen=[]
for length in (3,6,9):
    for pattern in itertools.product(range(3),repeat=length):
        if sum(pattern[i%length]==0 for i in range(105))<48:
            continue
        r=routine(pattern,31000)
        if r['success']:
            screen.append((r['total_cost'],r['steps'],pattern))
screen.sort()
cache={}
frontier=[]
rows=[]
for limit in range(28,36):
    candidates=[r for r in screen if r[1]<=limit*3][:50]
    best=None
    for _,_,pattern in candidates:
        if pattern not in cache:
            cache[pattern]=[routine(pattern,s) for s in range(32000,32020)]
        rr=cache[pattern]
        rate=mean(r['success'] and r['steps']<=limit*3 for r in rr)
        rank=(-rate,mean(r['total_cost'] for r in rr))
        if best is None or rank<best[0]:
            best=(rank,pattern)
    pattern=best[1]
    rr=[routine(pattern,s) for s in range(2000000,2000100)]
    record=dict(max_completion_day=limit,pattern=list(pattern),actions=[ACTIONS[a] for a in pattern],
                within_limit_rate=mean(r['success'] and r['steps']<=limit*3 for r in rr),**metrics(rr))
    frontier.append(record)
    for seed,r in zip(range(2000000,2000100),rr):
        rows.append(dict(max_completion_day=limit,eval_seed=seed,completed=int(r['success']),
                         within_limit=int(r['success'] and r['steps']<=limit*3),
                         on_time=int(r['on_time']),steps=r['steps'],total_cost=r['total_cost'],
                         final_mentor=r['final_mentor'],final_mood=r['final_mood']))
    print(record,flush=True)
p=ROOT/'results_grace'
(p/'fixed_frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2),encoding='utf-8')
with (p/'fixed_frontier.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

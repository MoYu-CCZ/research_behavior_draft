"""Re-search and retrain under slot-specific work/rest assumptions."""
import argparse,csv,itertools,json
from pathlib import Path
from statistics import mean
from agent import QLearningAgent
from environment import ResearchEnv,simulate_one_episode,train_agent
from config import ACTIONS,LAB_WORK_BY_SLOT,DORM_ENERGY_BY_SLOT,DORM_MOOD_BY_SLOT
from search_plans import metrics
from visualization import save_plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_time_aware_v3'
OPTIONS=dict(grace_days=7,delay_mood_penalty=2,reward_mode='cost',state_mode='detailed',model_profile='time_aware')

def run(plan,seed,periodic=True,profile='time_aware',penalty=2):
    options=OPTIONS|dict(model_profile=profile,delay_mood_penalty=penalty)
    e=ResearchEnv(4,seed,total_work=48,**options);e.reset()
    while not e.done:
        a=plan[e.t%len(plan)] if periodic else plan[e.t]
        if a not in e.allowed_actions():a=1
        e.step(a)
    return e.result()

def score(rr,ontime=False):
    return (-mean(r['success'] and (r['on_time'] or not ontime) for r in rr),
            mean(r['total_cost'] for r in rr))

def aggregate(rr):
    data=metrics(rr)
    data['mean_final_mentor']=mean(r['final_mentor'] for r in rr)
    data['mean_final_mood']=mean(r['final_mood'] for r in rr)
    data['action_distribution']={}
    for slot,name in enumerate(['morning','afternoon','evening']):
        counts={a:0 for a in ACTIONS}
        for r in rr:
            for step in r['slot_logs']:
                if (step['step']-1)%3==slot:counts[step['action']]+=1
        total=sum(counts.values())
        data['action_distribution'][name]={a:v/total for a,v in counts.items()}
    return data

def save_q(agent,seed):
    (OUT/('q_table_'+str(seed)+'.json')).write_text(json.dumps(dict(seed=seed,gamma=agent.gamma,
        alpha=agent.alpha,model_profile='time_aware',state_mode='detailed',
        lab_work=LAB_WORK_BY_SLOT,dorm_energy=DORM_ENERGY_BY_SLOT,dorm_mood=DORM_MOOD_BY_SLOT,
        q_table=[dict(state=list(k),values=v) for k,v in agent.q_table.items()])),encoding='utf-8')

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--replay',choices=['fixed','simple','rl']);parser.add_argument('--seed',type=int,default=4000000)
    parser.add_argument('--outdir',default='results_time_aware_v3')
    args=parser.parse_args();OUT=ROOT/args.outdir;OUT.mkdir(exist_ok=True)
    if args.replay:
        settings=json.loads((OUT/'settings.json').read_text(encoding='utf-8'))
        if settings['dorm_energy'] != list(DORM_ENERGY_BY_SLOT) or settings['dorm_mood'] != list(DORM_MOOD_BY_SLOT):
            raise ValueError('Saved experiment parameters differ from current config; do not replay incompatible Q tables.')
        if args.replay=='simple':
            data=json.loads((OUT/'selected_plan.json').read_text(encoding='utf-8'))
            r=run(data['periods']['ontime'],args.seed,True)
        elif args.replay=='fixed':
            plan=json.loads((OUT/'selected_plan.json').read_text(encoding='utf-8'))['ontime_plan']
            r=run(plan,args.seed,False)
        else:
            d=json.loads((OUT/'q_table_42.json').read_text(encoding='utf-8'))
            agent=QLearningAgent(ACTIONS,gamma=d['gamma'],alpha=d['alpha'],seed=42)
            agent.q_table={tuple(x['state']):x['values'] for x in d['q_table']}
            r=simulate_one_episode(agent,seed=args.seed,weeks=4,total_work=48,**OPTIONS)
        for row in r['logs']:print(row)
        print({k:v for k,v in r.items() if k not in ('logs','slot_logs')});return
    screen=[]
    for length in (3,6,9):
        for pattern in itertools.product(range(3),repeat=length):
            if sum(round(LAB_WORK_BY_SLOT[i%3]*10) for i in range(105) if pattern[i%length]==0)<480:continue
            r=run(pattern,41000)
            if r['success']:screen.append((r['total_cost'],pattern,r['on_time']))
    screen.sort();dev=list(range(42000,42020));cache={}
    chosen={}
    for label,ontime in [('any',False),('ontime',True)]:
        shortlist=[x for x in screen if x[2] or not ontime][:40]
        ranked=[]
        for _,pattern,_ in shortlist:
            if pattern not in cache:cache[pattern]=[run(pattern,s) for s in dev]
            ranked.append((score(cache[pattern],ontime),pattern))
        ranked.sort();chosen[label]=ranked[0][1]
    plan=[chosen['ontime'][i%len(chosen['ontime'])] for i in range(105)]
    # Development-only daily reorder search, preserving daily action counts.
    for day in range(28):
        start=day*3;best=(score([run(plan,s,False) for s in dev],True),plan)
        for actions in set(itertools.permutations(plan[start:start+3])):
            trial=plan.copy();trial[start:start+3]=actions
            value=score([run(trial,s,False) for s in dev],True)
            if value<best[0]:best=(value,trial)
        plan=best[1]
    (OUT/'selected_plan.json').write_text(json.dumps(dict(ontime_plan=plan,periods=chosen,
        screened_successful=len(screen),development_seeds=dev),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Selected fixed plans',chosen,flush=True)
    eval_seeds=list(range(4000000,4000100));summary={};rows=[]
    def record(label,rr,training_seed=None,penalty=2):
        for s,r in zip(eval_seeds,rr):
            rows.append(dict(policy=label,training_seed=training_seed,eval_seed=s,delay_penalty=penalty,
                completed=int(r['success']),on_time=int(r['on_time']),steps=r['steps'],
                total_cost=r['total_cost'],final_mentor=r['final_mentor'],final_mood=r['final_mood'],**r['costs']))
    old=json.loads((ROOT/'results_grace'/'fixed_ontime_plan.json').read_text(encoding='utf-8'))['plan']
    previous=json.loads((ROOT/'results_time_aware'/'selected_plan.json').read_text(encoding='utf-8'))['ontime_plan']
    policies=[('previous_time_aware_fixed',previous,False),('original_fixed',(0,0,1),True),('old_optimized_fixed',old,False),
              ('simple_no_afternoon_rest',chosen['ontime'],True),('new_ontime_fixed',plan,False),('new_min_cost_fixed',chosen['any'],True),
              ('morning_dorm_fixed',(1,0,0),True)]
    for label,p,periodic in policies:
        rr=[run(p,s,periodic) for s in eval_seeds];summary[label]=aggregate(rr);record(label,rr)
        if label=='new_ontime_fixed':
            (OUT/'fixed_example.json').write_text(json.dumps(rr[0],ensure_ascii=False,indent=2),encoding='utf-8')
            save_plots(rr[0],48,4,OUT/'fixed_plots')
    for penalty in (1,4):
        rr=[run(chosen['any'],s,True,penalty=penalty) for s in eval_seeds]
        summary['min_cost_penalty_'+str(penalty)]=aggregate(rr);record('new_min_cost_fixed',rr,penalty=penalty)
    combined=[];per_seed=[]
    for seed in (42,43,44):
        agent=QLearningAgent(ACTIONS,gamma=1,epsilon=0.1,seed=seed)
        for ep in range(2000):
            e=ResearchEnv(4,seed*100000+ep,total_work=48,**OPTIONS);obs=e.reset()
            while not e.done:
                a=plan[e.t]
                if a not in e.allowed_actions():a=1
                nxt,reward,terminated,truncated,_=e.step(a)
                agent.learn(obs,a,reward,nxt,terminated or truncated,e.allowed_actions());obs=nxt
        train_agent(agent,10000,decay_interval=500,decay_rate=0.8,min_epsilon=0.005,
            seed=seed*100000+2000,weeks=4,total_work=48,**OPTIONS)
        rr=[simulate_one_episode(agent,seed=s,weeks=4,total_work=48,**OPTIONS) for s in eval_seeds]
        record('new_warm_rl',rr,seed);combined.extend(rr);per_seed.append(dict(seed=seed,**aggregate(rr)))
        save_q(agent,seed);print('RL',seed,metrics(rr),flush=True)
        if seed==42:
            (OUT/'rl_example.json').write_text(json.dumps(rr[0],ensure_ascii=False,indent=2),encoding='utf-8')
            save_plots(rr[0],48,4,OUT/'rl_plots')
    summary['new_warm_rl']=dict(**aggregate(combined),per_seed=per_seed)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'evaluation.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'schedule.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['day','morning','afternoon','evening'])
        for d in range(35):w.writerow([d+1]+[ACTIONS[a] for a in plan[d*3:d*3+3]])
    (OUT/'settings.json').write_text(json.dumps(dict(options=OPTIONS,target_days=28,max_days=35,
        work=48,lab_work=LAB_WORK_BY_SLOT,dorm_energy=DORM_ENERGY_BY_SLOT,dorm_mood=DORM_MOOD_BY_SLOT,
        evening_lab_recovery_penalty=2,train_seeds=[42,43,44],demo_episodes=2000,online_episodes=10000,
        evaluation_seeds=eval_seeds),ensure_ascii=False,indent=2),encoding='utf-8')
    names=[x[0] for x in policies]+['new_warm_rl']
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,key in zip(axes,['mean_cost','on_time_rate','mean_days']):
        ax.bar(range(len(names)),[summary[n][key] or 0 for n in names]);ax.set(xticks=range(len(names)),xticklabels=names,ylabel=key);ax.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for slot,ax in zip(['morning','afternoon','evening'],axes):
        bottom=[0]*len(names)
        for action in ACTIONS:
            values=[summary[n]['action_distribution'][slot][action] for n in names]
            ax.bar(range(len(names)),values,bottom=bottom,label={'实验室':'Lab','宿舍':'Dorm','健身房':'Gym'}[action]);bottom=[x+y for x,y in zip(bottom,values)]
        ax.set(title=slot,xticks=range(len(names)),xticklabels=names);ax.tick_params(axis='x',rotation=35);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'action_distribution.png',dpi=160);plt.close(fig)
    print(json.dumps({n:{k:v for k,v in m.items() if k not in ('action_distribution','per_seed')} for n,m in summary.items()},indent=2),flush=True)

if __name__=='__main__':main()







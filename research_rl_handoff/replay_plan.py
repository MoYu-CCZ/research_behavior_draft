"""Replay selected fixed schedules or learned Q policies."""
import argparse,json
from pathlib import Path
from agent import QLearningAgent
from config import ACTIONS
from search_plans import routine
from environment import ResearchEnv,simulate_one_episode
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--policy',choices=['fixed','on_time','one_day','rl','warm_rl'],default='on_time')
    p.add_argument('--seed',type=int,default=3000000)
    p.add_argument('--training-seed',type=int,default=42)
    p.add_argument('--delay-mood-penalty',type=float,default=2)
    a=p.parse_args();out=ROOT/'results_grace'
    if a.policy=='fixed':
        data=json.loads((out/'fixed_search.json').read_text(encoding='utf-8'))
        result=routine(data['selected_pattern'],a.seed,a.delay_mood_penalty)
    elif a.policy=='one_day':
        data=json.loads((out/'fixed_frontier.json').read_text(encoding='utf-8'))[1]
        result=routine(data['pattern'],a.seed,a.delay_mood_penalty)
    elif a.policy=='on_time':
        plan=json.loads((out/'fixed_ontime_plan.json').read_text(encoding='utf-8'))['plan']
        env=ResearchEnv(4,a.seed,total_work=48,grace_days=7,delay_mood_penalty=a.delay_mood_penalty)
        env.reset()
        while not env.done:
            action=plan[env.t]
            if action not in env.allowed_actions():action=1
            env.step(action)
        result=env.result()
    else:
        prefix='warm_q_table_' if a.policy=='warm_rl' else 'q_table_'
        data=json.loads((out/(prefix+str(a.training_seed)+'.json')).read_text(encoding='utf-8'))
        agent=QLearningAgent(ACTIONS,gamma=data['gamma'],alpha=data['alpha'],seed=data['seed'])
        agent.q_table={tuple(r['state']):r['values'] for r in data['q_table']}
        result=simulate_one_episode(agent,seed=a.seed,weeks=4,total_work=48,grace_days=7,
                 delay_mood_penalty=a.delay_mood_penalty,reward_mode='cost',
                 state_mode=data.get('state_mode','compact'))
    for row in result['logs']:
        print('Day',row['day'],row['actions'],'energy',row['energy'],'mood',row['mood'],
              'anxiety',row['anxiety'],'remaining',row['work_remaining'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('logs','slot_logs')},ensure_ascii=False,indent=2))
    (out/('replay_'+a.policy+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()

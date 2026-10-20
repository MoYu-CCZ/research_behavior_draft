"""Validate final parameters, aggregates and representative saved-policy replays."""
from pathlib import Path
import csv, json, math, re, sys
from statistics import mean
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'research_rl_handoff'
sys.path.insert(0, str(SOURCE))
from agent import QLearningAgent
from config import ACTIONS, LAB_WORK_BY_SLOT, DORM_ENERGY_BY_SLOT, DORM_MOOD_BY_SLOT
from environment import ResearchEnv, simulate_one_episode
DATA = SOURCE / 'results_time_aware_v3'
def read(name):
    return json.loads((DATA / name).read_text(encoding='utf-8-sig'))
def check(condition, message):
    if not condition:
        raise AssertionError(message)
def close(a, b, label):
    check(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9), label)
settings, summary, plan = read('settings.json'), read('summary.json'), read('selected_plan.json')
for key, expected in [('lab_work', LAB_WORK_BY_SLOT), ('dorm_energy', DORM_ENERGY_BY_SLOT), ('dorm_mood', DORM_MOOD_BY_SLOT)]:
    check(settings[key] == list(expected), key + ': parameters')
check((settings['work'], settings['target_days'], settings['max_days']) == (48, 28, 35), 'task bounds')
check(settings['options']['delay_mood_penalty'] == 2, 'delay penalty')
with (DATA / 'evaluation.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))
fixed = {'original_fixed': ((0,0,1), True), 'morning_dorm_fixed': ((1,0,0), True),
         'simple_no_afternoon_rest': (plan['periods']['ontime'], True),
         'new_ontime_fixed': (plan['ontime_plan'], False),
         'new_min_cost_fixed': (plan['periods']['any'], True)}
for policy in list(fixed) + ['new_warm_rl']:
    selected = [r for r in rows if r['policy'] == policy and r['delay_penalty'] == '2']
    check(len(selected) == (300 if policy == 'new_warm_rl' else 100), policy + ': count')
    check({int(r['eval_seed']) for r in selected} == set(settings['evaluation_seeds']), policy + ': seeds')
    for column, metric in [('completed','completion_rate'), ('on_time','on_time_rate'),
                           ('total_cost','mean_cost'), ('final_mood','mean_final_mood'), ('final_mentor','mean_final_mentor')]:
        close(mean(float(r[column]) for r in selected), summary[policy][metric], policy + ': ' + metric)
ENV_SEED = 4000000
def check_replay(policy, result, training_seed=None):
    selected = [r for r in rows if r['policy'] == policy and int(r['eval_seed']) == ENV_SEED
                and r['delay_penalty'] == '2' and (training_seed is None or r['training_seed'] == str(training_seed))]
    check(len(selected) == 1, policy + ': representative row')
    row = selected[0]
    check(result['steps'] == int(row['steps']), policy + ': steps')
    check(result['success'] and result['completion'] == 1, policy + ': completion')
    for key in ('total_cost','final_mood','final_mentor'):
        close(result[key], row[key], policy + ': ' + key)
for policy, (actions, periodic) in fixed.items():
    env = ResearchEnv(4, ENV_SEED, total_work=48, **settings['options'])
    env.reset()
    while not env.done:
        action = actions[env.t % len(actions)] if periodic else actions[env.t]
        env.step(action if action in env.allowed_actions() else 1)
    check_replay(policy, env.result())
for seed in settings['train_seeds']:
    saved = read('q_table_' + str(seed) + '.json')
    check(saved['dorm_mood'] == settings['dorm_mood'], 'Q table parameters')
    agent = QLearningAgent(ACTIONS, alpha=saved['alpha'], gamma=saved['gamma'], seed=seed)
    agent.q_table = {tuple(r['state']): r['values'] for r in saved['q_table']}
    result = simulate_one_episode(agent, seed=ENV_SEED, weeks=4, total_work=48, **settings['options'])
    check_replay('new_warm_rl', result, seed)
print('PASS: final parameters, six-strategy aggregates, five fixed-policy and three Q-table representative replays.')
PAPER = ROOT / 'paper' if (ROOT / 'paper' / 'research_behavior_draft_v3.tex').is_file() else ROOT
tex = (PAPER / 'research_behavior_draft_v3.tex').read_text(encoding='utf-8-sig')
missing = [name for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)
           if not (PAPER / name).is_file()]
print('Paper assets pending: ' + ', '.join(sorted(set(missing))) if missing else 'All referenced paper assets present.')

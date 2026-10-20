"""Replay final V3 strategies and plot six actual schedules without changing the paper."""
from pathlib import Path
import sys,json,csv
HERE=Path(__file__).resolve().parent
PAPER=HERE.parent.parent
PROJECT=PAPER if (PAPER/'research_rl_handoff').is_dir() else PAPER.parent
sys.path.insert(0,str(PAPER/'.figure_deps'))
sys.path.insert(0,str(PROJECT/'research_rl_handoff'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib import font_manager
from environment import ResearchEnv,simulate_one_episode
from agent import QLearningAgent
from config import ACTIONS,LAB_WORK_BY_SLOT,DORM_ENERGY_BY_SLOT,DORM_MOOD_BY_SLOT
font_path=Path('C:/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font_path)).get_name(),
                     'axes.unicode_minus':False,'svg.fonttype':'path','hatch.linewidth':0.7})
DATA=PROJECT/'research_rl_handoff'/'results_time_aware_v3'
settings=json.loads((DATA/'settings.json').read_text(encoding='utf-8'))
assert settings['lab_work']==list(LAB_WORK_BY_SLOT)
assert settings['dorm_energy']==list(DORM_ENERGY_BY_SLOT)
assert settings['dorm_mood']==list(DORM_MOOD_BY_SLOT)
plan=json.loads((DATA/'selected_plan.json').read_text(encoding='utf-8'))
ENV_SEED=4000000
TRAIN_SEED=42

def replay(actions,periodic=True):
    env=ResearchEnv(4,ENV_SEED,total_work=48,**settings['options']);env.reset()
    while not env.done:
        action=actions[env.t%len(actions)] if periodic else actions[env.t]
        if action not in env.allowed_actions(): action=1
        env.step(action)
    return env.result()

q=json.loads((DATA/'q_table_42.json').read_text(encoding='utf-8'))
assert q['dorm_mood']==settings['dorm_mood']
agent=QLearningAgent(ACTIONS,alpha=q['alpha'],gamma=q['gamma'],seed=TRAIN_SEED)
agent.q_table={tuple(row['state']):row['values'] for row in q['q_table']}
policies=[('a','original_fixed','常规固定作息',replay((0,0,1))),
          ('b','morning_dorm_fixed','上午补觉',replay((1,0,0))),
          ('c','simple_no_afternoon_rest','简单三天循环',replay(plan['periods']['ontime'])),
          ('d','new_ontime_fixed','优化固定日程',replay(plan['ontime_plan'],False)),
          ('e','new_warm_rl','示范辅助Q-learning',simulate_one_episode(agent,seed=ENV_SEED,weeks=4,total_work=48,**settings['options'])),
          ('f','new_min_cost_fixed','延期低成本循环',replay(plan['periods']['any']))]
# Verify the replays against the recorded evaluation rows, including RL training seed.
with (DATA/'evaluation.csv').open(encoding='utf-8-sig',newline='') as f:
    recorded=list(csv.DictReader(f))
for letter,key,title,result in policies:
    rows=[r for r in recorded if r['policy']==key and int(r['eval_seed'])==ENV_SEED and r['delay_penalty']=='2'
          and (key!='new_warm_rl' or r['training_seed']==str(TRAIN_SEED))]
    assert len(rows)==1,key
    assert result['steps']==int(rows[0]['steps']),key
    assert abs(result['total_cost']-float(rows[0]['total_cost']))<1e-9,key
    assert result['success'] and result['completion']==1,key

plt.rcParams.update({'text.color':'#183650','axes.labelcolor':'#183650',
                     'xtick.color':'#38536B','ytick.color':'#38536B'})
STYLE={'实验室':dict(facecolor='#174A78',edgecolor='#EAF3FA',hatch=None),
       '宿舍':dict(facecolor='#6BA6CF',edgecolor='#EAF3FA',hatch=None),
       '健身房':dict(facecolor='#C9DFF0',edgecolor='#A0BDD3',hatch=None)}
legend=[Patch(facecolor='#174A78',edgecolor='#174A78',label='实验室'),
        Patch(facecolor='#6BA6CF',edgecolor='#6BA6CF',label='宿舍'),
        Patch(facecolor='#C9DFF0',edgecolor='#A0BDD3',label='健身房'),
        Patch(facecolor='white',edgecolor='#CAD8E3',label='结束后未执行'),
        Line2D([0],[0],marker='*',markerfacecolor='white',markeredgecolor='#183650',color='none',markersize=11,label='完成时段')]
slots=['上午','下午','晚上']

def draw(ax,letter,title,result):
    ax.set_xlim(0,35);ax.set_ylim(3,0)
    for day in range(35):
        for slot in range(3):
            ax.add_patch(Rectangle((day,slot),1,1,facecolor='white',edgecolor='#D8E4ED',lw=.35))
    for row in result['slot_logs']:
        step=row['step']-1;day,slot=divmod(step,3)
        ax.add_patch(Rectangle((day,slot),1,1,lw=.4,**STYLE[row['action']]))
    last=result['steps']-1;last_day,last_slot=divmod(last,3)
    ax.plot(last_day+.5,last_slot+.5,marker='*',markersize=8.3,markerfacecolor='white',markeredgecolor='white',markeredgewidth=.4,zorder=6)
    # Deadline is the boundary after day 28, not the center of that day's cells.
    ax.axvline(28,color='#183650',lw=1.35,ls=(0,(4,2)),zorder=7).set_path_effects([pe.Stroke(linewidth=2.4,foreground='white'),pe.Normal()])
    ax.annotate('28天目标',xy=(28,-.025),xytext=(28,-.28),ha='center',va='bottom',fontsize=9,
                xycoords=('data','data'),textcoords=('data','data'),annotation_clip=False)
    ax.set_xticks([.5,6.5,13.5,20.5,27.5,34.5],['1','7','14','21','28','35'])
    ax.set_yticks([.5,1.5,2.5],slots)
    ax.tick_params(axis='both',length=0,labelsize=10.5,pad=4)
    for spine in ax.spines.values():spine.set_linewidth(.65);spine.set_color('#7895AB')
    ax.set_title(f'({letter}) {title}',loc='left',fontsize=13.0,pad=25,fontweight='bold')
    status='按时' if result['on_time'] else '延期7天'
    ax.set_xlabel(f'第{last_day+1}天{slots[last_slot]}完成（{status}）',fontsize=10.4,labelpad=7)

fig=plt.figure(figsize=(14,6.2),facecolor='white')
for i,(letter,key,title,result) in enumerate(policies):
    row,col=divmod(i,3)
    ax=fig.add_axes([.056+col*.328,.585-row*.375,.272,.20])
    draw(ax,letter,title,result)
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.52,.055),ncol=5,frameon=False,fontsize=11,
           handlelength=1.7,columnspacing=1.65)
fig.text(.5,.032,'横轴：任务第几天；目标线位于第28天结束处。仅显示已执行动作，完成后留白。',ha='center',fontsize=10)
fig.text(.5,.006,'示例环境种子4000000；RL训练种子42。此图为单回合轨迹，不是100个环境的统计汇总。',ha='center',fontsize=10)
fig.savefig(HERE/'six_strategy_schedule_heatmaps.png',dpi=400)
fig.savefig(HERE/'six_strategy_schedule_heatmaps.svg')
plt.close(fig)

for letter,key,title,result in policies:
    fig=plt.figure(figsize=(8,3.25),facecolor='white')
    ax=fig.add_axes([.10,.37,.85,.34]);draw(ax,letter,title,result)
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.52,.105),ncol=5,frameon=False,fontsize=9.5,
               columnspacing=1.1,handlelength=1.4)
    fig.text(.5,.035,'环境种子4000000；RL训练种子42。白色空格为回合结束后未执行时段。',ha='center',fontsize=9.4)
    fig.savefig(HERE/f'fig2{letter}_{key}.png',dpi=400)
    fig.savefig(HERE/f'fig2{letter}_{key}.svg')
    plt.close(fig)

with (HERE/'executed_schedule.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['policy','environment_seed','training_seed','step','day','slot','action','completed_here'])
    writer.writeheader()
    for letter,key,title,result in policies:
        for row in result['slot_logs']:
            day,slot=divmod(row['step']-1,3)
            writer.writerow(dict(policy=key,environment_seed=ENV_SEED,training_seed=TRAIN_SEED if key=='new_warm_rl' else '',
                                 step=row['step'],day=day+1,slot=slots[slot],action=row['action'],completed_here=int(row['step']==result['steps'])))
summary=[dict(policy=key,title=title,environment_seed=ENV_SEED,training_seed=TRAIN_SEED if key=='new_warm_rl' else None,
              completed_steps=r['steps'],completion=r['completion'],on_time=r['on_time'],cost=r['total_cost']) for _,key,title,r in policies]
(HERE/'replay_metadata.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))


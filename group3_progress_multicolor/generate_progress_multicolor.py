"""Replay final V3 strategies and plot six actual schedules without changing the paper."""
from pathlib import Path
import sys,json,csv
HERE=Path(__file__).resolve().parent
PAPER=HERE.parent.parent
PROJECT=PAPER.parent
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

plt.rcParams.update({'text.color':'#222222','axes.labelcolor':'#222222',
                     'xtick.color':'#333333','ytick.color':'#333333'})
# Weekly endpoint samples plus the exact completion instant; no post-termination extrapolation.
colors=['#0072B2','#E69F00','#CC79A7','#009E73','#D55E00','#56B4E9']
styles=['-','--',':','-','--','-.']
markers=['o','s','^','o','x','D']
weekly=[]
for letter,key,title,result in policies:
    work=0
    samples=[dict(step=0,week=0.0,completed_work=0.0)]
    for row in result['slot_logs']:
        work=round(work+row['work_progress'],10)
        if row['step']%21==0 or row['step']==result['steps']:
            samples.append(dict(step=row['step'],week=row['step']/21,completed_work=work))
    assert abs(work-48)<1e-9
    weekly.append(dict(policy=key,title=title,samples=samples))
assert [s['completed_work'] for s in weekly[3]['samples']]==[s['completed_work'] for s in weekly[4]['samples']]
assert [s['step'] for s in weekly[3]['samples']]==[s['step'] for s in weekly[4]['samples']]
fig,ax=plt.subplots(figsize=(10,6.7),facecolor='white')
fig.subplots_adjust(left=.095,right=.975,top=.895,bottom=.285)
for i,policy in enumerate(weekly):
    x=[s['week'] for s in policy['samples']]
    y=[s['completed_work'] for s in policy['samples']]
    ax.plot(x,y,color=colors[i],linestyle=styles[i],lw=[1.9,1.9,4.2,2.7,1.5,1.9][i],
            marker=markers[i],markersize=9 if i==2 else 6.3,markerfacecolor='white' if i in (2,3) else colors[i],
            markeredgewidth=1.2,label=policy['title'],zorder=4+i)
    ax.plot(x[-1],y[-1],marker='o',markersize=8.5 if i==3 else 5.5,
            markerfacecolor='white' if i==3 else colors[i],markeredgecolor=colors[i],markeredgewidth=1.5,zorder=12+i)
ax.plot([0,4,5],[0,48,48],color='#404040',ls=(0,(6,3)),lw=1.65,
        label='期望进度：4周完成48单位',zorder=2)
ax.axvline(4,color='#718293',lw=1,ls=(0,(3,3)),zorder=1)
ax.axvline(5,color='#A5B1BC',lw=.9,ls=':',zorder=1)
ax.text(4.035,3,'28天目标',rotation=90,color='#555555',fontsize=10,va='bottom')
ax.text(4.95,3,'35天上限',rotation=90,color='#718293',fontsize=10,va='bottom',ha='right')
ax.annotate('循环、优化及RL的前三周\n周末进度重合',xy=(2,25.4),xytext=(1.1,38),
            fontsize=10.5,color='#444444',ha='center',
            arrowprops=dict(arrowstyle='->',color='#718293',lw=.8),
            bbox=dict(facecolor='white',edgecolor='none',pad=2))
ax.set(xlim=(0,5.08),ylim=(0,51),xticks=range(6),yticks=[0,12,24,36,48],
       xlabel='经过时间（周；每周21个时段）',ylabel='累计完成工作量（单位）')
ax.set_title('六种策略的每周实际进度与期望进度',fontsize=16,fontweight='bold',pad=14,color='#222222')
ax.tick_params(axis='both',labelsize=11)
ax.xaxis.label.set_size(12);ax.yaxis.label.set_size(12)
ax.grid(axis='y',color='#E2E9EF',linewidth=.7)
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color('#8397A7')
handles,labels=ax.get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.093),ncol=3,frameon=False,
           fontsize=10.5,columnspacing=1.6,handlelength=3.1)
fig.text(.5,.062,'仅连接周末与实际完成时点；曲线在完成后停止。优化固定日程与RL的轨迹完全重合。',
         ha='center',fontsize=10,color='#555555')
fig.text(.5,.032,'示例环境种子4000000；RL训练种子42。期望进度在第4周达到48，延期期间不继续增加。',
         ha='center',fontsize=10,color='#555555')
fig.savefig(HERE/'six_strategy_weekly_progress.png',dpi=400)
fig.savefig(HERE/'six_strategy_weekly_progress.svg')
plt.close(fig)
(HERE/'weekly_progress_data.json').write_text(json.dumps(weekly,ensure_ascii=False,indent=2),encoding='utf-8')
with (HERE/'weekly_progress_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['policy','title','step','week','completed_work','expected_work'])
    writer.writeheader()
    for policy in weekly:
        for sample in policy['samples']:
            writer.writerow(dict(policy=policy['policy'],title=policy['title'],**sample,
                                 expected_work=min(48,12*sample['week'])))
print('Saved weekly progress comparison for six strategies, including exact completion instants.')


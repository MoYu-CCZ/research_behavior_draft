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
from matplotlib.ticker import PercentFormatter
import numpy as np
colors=['#0072B2','#E69F00','#CC79A7','#009E73','#D55E00','#56B4E9']
styles=['-','--',':','-','--','-.']
markers=['o','s','^','o','x','D']
# Re-execute the actual action sequence to obtain post-settlement mentor and anxiety
# at every slot. This does not choose or alter actions or retrain a policy.
traces=[]
for letter,key,title,result in policies:
    env=ResearchEnv(4,ENV_SEED,total_work=48,**settings['options']);env.reset()
    rows=[dict(step=0,days=0,energy=env.energy,mood=env.mood,mentor=env.mentor,anxiety=env.anxiety,fatigue=0)]
    for log in result['slot_logs']:
        env.step(ACTIONS.index(log['action']))
        assert env.energy==log['energy']
        assert env.mood==log['mood']
        assert abs(env.anxiety/100-log['anxiety'])<1e-9
        rows.append(dict(step=env.t,days=env.t/3,energy=env.energy,mood=env.mood,
                         mentor=env.mentor,anxiety=env.anxiety,fatigue=(100-max(0,env.energy))/100))
    assert env.done and env.t==result['steps'] and env.mentor==result['final_mentor']
    assert abs(sum(env.cost.values())-result['total_cost'])<1e-9
    traces.append(dict(policy=key,title=title,rows=rows))
assert traces[3]['rows']==traces[4]['rows']
legendhandles=[]

def decorate(ax,title,ylabel,ylim=None):
    ax.set_xlim(0,35.5)
    ax.set_xticks([0,7,14,21,28,35])
    if ylim is not None:ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel,fontsize=12)
    ax.set_title(title,fontsize=15.5,fontweight='bold',pad=13,color='#222222')
    ax.grid(axis='y',color='#E5E5E5',lw=.65)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#777777')
    ax.tick_params(labelsize=10.5)
    ax.axvline(28,color='#666666',lw=.85,ls=(0,(4,3)),zorder=1)
    ax.axvline(35,color='#999999',lw=.7,ls=':',zorder=1)
    ax.text(28.35,.025,'28天目标',transform=ax.get_xaxis_transform(),rotation=90,fontsize=9,color='#666666',va='bottom')
    ax.text(34.6,.025,'35天上限',transform=ax.get_xaxis_transform(),rotation=90,fontsize=9,color='#777777',va='bottom',ha='right')

def curves(ax,key):
    handles=[]
    for i,trace in enumerate(traces):
        rows=trace['rows'];x=[r['days'] for r in rows];y=[r[key] for r in rows]
        # Weekly/terminal markers supplement line styles; all slot values remain plotted.
        indices=[j for j,r in enumerate(rows) if r['step']%21==0 or j==len(rows)-1]
        h,=ax.plot(x,y,color=colors[i],ls=styles[i],lw=[1.45,1.45,3.6,2.5,1.2,1.5][i],
                  drawstyle='steps-post' if key=='mentor' else 'default',marker=markers[i],markersize=7.8 if i==2 else 5.4,markevery=indices,
                  markerfacecolor='white' if i in (2,3) else colors[i],markeredgewidth=1,
                  label=trace['title'],zorder=3+i)
        handles.append(h)
    return handles

def legend(fig,handles,y):
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,y),ncol=3,frameon=False,
               fontsize=10.5,columnspacing=1.5,handlelength=3)

def output(fig,name):
    fig.savefig(HERE/(name+'.png'),dpi=400,facecolor='white')
    fig.savefig(HERE/(name+'.svg'),facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(10,6.7),facecolor='white')
fig.subplots_adjust(left=.095,right=.97,top=.9,bottom=.27)
fatigue_max=max(r['fatigue'] for tr in traces for r in tr['rows'])
decorate(ax,'六种策略的疲劳指标变化','精力亏空比例  (100 − 精力) / 100',(-.005,.205))
handles=curves(ax,'fatigue');ax.set_yticks([0,.05,.10,.15,.20]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax.set_xlabel('经过时间（天；每个时段 = 1/3天）',fontsize=12)
legend(fig,handles,.105)
fig.text(.5,.06,'逐时段结算后的精力亏空；晚间包括自然恢复。曲线在完成时停止，不进行平滑。',ha='center',fontsize=10,color='#555555')
fig.text(.5,.03,'环境种子4000000；RL训练种子42。优化固定日程与RL重合；该指标不是临床疲劳测量。',ha='center',fontsize=10,color='#555555')
output(fig,'six_strategy_fatigue_changes')

fig,axes=plt.subplots(2,1,figsize=(10,9),sharex=True,facecolor='white')
fig.subplots_adjust(left=.095,right=.97,top=.87,bottom=.205,hspace=.32)
decorate(axes[0],'(a) 心情变化','心情（0–100）',(-3,103))
decorate(axes[1],'(b) 焦虑变化','焦虑',(-.5,20))
handles=curves(axes[0],'mood');curves(axes[1],'anxiety')
axes[0].set_yticks([0,20,40,60,80,100]);axes[1].set_yticks([0,5,10,15,20])
axes[1].set_xlabel('经过时间（天；每个时段 = 1/3天）',fontsize=12)
fig.suptitle('六种策略的心理状态变化',fontsize=16,fontweight='bold',y=.98)
legend(fig,handles,.075)
fig.text(.5,.043,'逐时段结算后的模拟状态；焦虑纵轴显示0–20，便于观察变化；完成时停止。',ha='center',fontsize=10,color='#555555')
fig.text(.5,.018,'环境种子4000000；RL训练种子42。优化固定日程与RL重合；单回合示例，不是总体均值。',ha='center',fontsize=10,color='#555555')
output(fig,'six_strategy_psychological_changes')

fig,ax=plt.subplots(figsize=(10,6.7),facecolor='white')
fig.subplots_adjust(left=.095,right=.97,top=.9,bottom=.27)
decorate(ax,'六种策略的导师满意度变化','导师满意度（0–100）',(-3,103))
ax.set_yticks([0,20,40,60,80,100]);handles=curves(ax,'mentor')
ax.set_xlabel('经过时间（天；每个时段 = 1/3天）',fontsize=12)
legend(fig,handles,.105)
fig.text(.5,.06,'逐时段结算后的满意度；变化来自导师检查和周末进度评价，曲线在完成时停止。',ha='center',fontsize=10,color='#555555')
fig.text(.5,.03,'环境种子4000000；RL训练种子42。优化固定日程与RL重合；单回合示例，不是总体均值。',ha='center',fontsize=10,color='#555555')
output(fig,'six_strategy_mentor_satisfaction')

with (HERE/'state_changes_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['policy','title','step','days','energy','fatigue','mood','anxiety','mentor'])
    writer.writeheader()
    for trace in traces:
        for row in trace['rows']:writer.writerow(dict(policy=trace['policy'],title=trace['title'],**row))
(HERE/'state_changes_data.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2),encoding='utf-8')
print('Generated fatigue, psychological-state and mentor-satisfaction plots; all use post-slot states.')


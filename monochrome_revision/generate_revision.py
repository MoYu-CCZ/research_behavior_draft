"""Monochrome paper diagrams: precise task clock and background line drawing."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent/'.figure_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle,Rectangle,Ellipse,Polygon,FancyArrowPatch,PathPatch
from matplotlib.path import Path as MplPath
FONT=Path('C:/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),
                     'axes.unicode_minus':False,'svg.fonttype':'path','font.size':13})

def setup(height=4.6,ymax=58):
    f=plt.figure(figsize=(8,height),facecolor='white')
    a=f.add_axes([0,0,1,1]);a.set(xlim=(0,100),ylim=(0,ymax));a.axis('off')
    return f,a

def label(a,x,y,s,size=13,ha='center',weight='normal'):
    return a.text(x,y,s,fontsize=size,ha=ha,va='center',color='black',fontweight=weight,linespacing=1.5)

def line(a,x,y,lw=1.5,ls='-'):
    a.plot(x,y,color='black',lw=lw,ls=ls,solid_capstyle='round',dash_capstyle='round')

def arrow(a,p,q,dashed=False):
    a.add_patch(FancyArrowPatch(p,q,arrowstyle='->',mutation_scale=13,
                               lw=1.35,color='black',linestyle='--' if dashed else '-'))

def save(f,name):
    f.canvas.draw()
    rend=f.canvas.get_renderer()
    for a in f.axes:
        for txt in a.texts:
            b=txt.get_window_extent(rend)
            if b.x0<0 or b.x1>f.bbox.width or b.y0<0 or b.y1>f.bbox.height:
                raise ValueError('Label outside canvas: '+txt.get_text())
    f.savefig(HERE/(name+'.png'),dpi=400,facecolor='white')
    f.savefig(HERE/(name+'.svg'),facecolor='white')
    plt.close(f)

# A: t counts completed slots. t=0 is before day 1, not day 1 itself.
f,a=setup()
label(a,3,54,'(a) 目标期限与延期上限',size=17,ha='left',weight='bold')
label(a,50,47.5,'任务总量48单位；每天3个时段；回合可在完成或失败时提前结束',size=12.6)
x=lambda t:8+87*t/105
# Interval brackets carry inequality-defined boundaries.
for left,right in [(0,84),(84,105)]:
    line(a,[x(left),x(right)],[36,36],lw=1)
    line(a,[x(left),x(left)],[34.5,36],lw=1)
    line(a,[x(right),x(right)],[34.5,36],lw=1)
label(a,(x(0)+x(84))/2,39.6,'按时完成区间：0 < t ≤ 84',size=13.3)
label(a,(x(84)+x(105))/2,39.6,'84 < t ≤ 105',size=11.5)
line(a,[x(0),x(84)],[28,28],lw=2)
line(a,[x(84),x(105)],[28,28],lw=2,ls='--')
for t in [0,21,42,63,84,105]:
    line(a,[x(t),x(t)],[27,29],lw=1.15)
    label(a,x(t),24,str(t),size=12)
    label(a,x(t),17,str(t//3),size=12)
label(a,3,24,'t',size=12)
label(a,3,17,'天',size=11)
line(a,[x(84),x(84)],[28.5,34],lw=.9,ls=':')
line(a,[x(105),x(105)],[28.5,34],lw=.9,ls=':')
label(a,x(84)-2,31.8,'目标',size=10.5,ha='right')
label(a,x(105)-2,31.8,'上限',size=10.5,ha='right')
label(a,50,10.5,'t：已执行时段数；下排为经过天数（第28天结束为t = 84）',size=11.6)
label(a,50,5,'延期日（第29—35天）：每天上午心情 −2；任务量保持不变',size=11.6)
save(f,'fig1a_task_timeline_bw')

# Background: person deciding where to go; food is contextual, not an RL action.
f,a=setup(4.8,60)
a.set_aspect('equal',adjustable='box')
# Center person with an open questioning gesture.
a.add_patch(Circle((50,34),3.4,fill=False,edgecolor='black',lw=1.8))
line(a,[50,50],[30.6,21],lw=1.8)
line(a,[50,46,42],[28.5,26,29],lw=1.8)
line(a,[50,54,58],[28.5,26,29],lw=1.8)
line(a,[50,46],[21,12.5],lw=1.8)
line(a,[50,54],[21,12.5],lw=1.8)
label(a,50,44.5,'去哪？',size=23)
# Dashed choice arrows, identical for all destinations.
arrow(a,(42,33),(29,43),True)
arrow(a,(58,33),(71,43),True)
arrow(a,(42,22),(31,16),True)
arrow(a,(58,22),(69,16),True)
# Cafeteria: rounded drumstick outline and a two-lobed bone.
verts=[(14,44),(11,47),(14,52),(19,52),(25,54),(29,50),(27,46),(26,42),(19,41),(14,44)]
codes=[MplPath.MOVETO]+[MplPath.CURVE4]*9
a.add_patch(PathPatch(MplPath(verts,codes),facecolor='white',edgecolor='black',lw=1.7))
line(a,[14,10],[44,40],lw=1.6)
line(a,[16,12],[42.5,38.5],lw=1.6)
a.add_patch(PathPatch(MplPath([(10,40),(7,42),(5.8,39),(8.5,37.8),(9,35.5),(12.5,36),(12,38.5)],
                           [MplPath.MOVETO]+[MplPath.CURVE4]*6),fill=False,edgecolor='black',lw=1.6))
a.add_patch(PathPatch(MplPath([(19,49),(23,50),(25,48),(24,46)],
                           [MplPath.MOVETO]+[MplPath.CURVE4]*3),fill=False,edgecolor='black',lw=1.0))
label(a,18,33.5,'食堂',size=15)
# Lab: laptop screen and trapezoidal keyboard/base.
a.add_patch(Rectangle((75,42),19,11.5,fill=False,edgecolor='black',lw=1.6))
a.add_patch(Rectangle((77,44),15,7.5,fill=False,edgecolor='black',lw=1))
a.add_patch(Polygon([(75,42),(94,42),(97,39),(72,39)],closed=True,fill=False,edgecolor='black',lw=1.6))
line(a,[81,88],[40.5,40.5],lw=1)
label(a,84.5,33.5,'实验室',size=15)
# Dorm: bed frame, pillow, cover and sleep symbol.
line(a,[5,5],[8,23],lw=1.7)
line(a,[5,29],[12.5,12.5],lw=1.7)
line(a,[29,29],[8,17],lw=1.7)
line(a,[5,29],[17,17],lw=1.6)
a.add_patch(Ellipse((10,18.4),6,2.8,fill=False,edgecolor='black',lw=1.4))
line(a,[15,15,29],[17,20,20],lw=1.5)
line(a,[29,29],[20,17],lw=1.5)
label(a,22,26,'Zzz',size=20)
label(a,17,4,'宿舍',size=15)
# Gym: dumbbell shaft and weight plates.
line(a,[77,92],[16,16],lw=2)
for xx,yy,ww,hh in [(73,11,3,10),(76,13,3,6),(91,13,3,6),(94,11,3,10)]:
    a.add_patch(Rectangle((xx,yy),ww,hh,facecolor='white',edgecolor='black',lw=1.6))
label(a,85,4,'健身房',size=15)
save(f,'fig0_background_choices')
print('Saved monochrome timeline and background line drawing (PNG + SVG).')



# 黑白图与伪代码修订

- fig0_background_choices.png／.svg：小人为主体，四个方向分别是鸡腿（食堂）、笔记本电脑（实验室）、床和Zzz（宿舍）、哑铃（健身房）。适合放在引言作为第一张图；食堂是背景意象，不是当前模型的动作。
- fig1a_task_timeline_bw.png／.svg：黑白时间轴，实线为目标期，虚线为有限延期；t明确表示已执行时段数，84／105分别为目标截止与硬截止，下排为经过天数。
- 图C不再作为图片使用，结算逻辑改为正文算法；独立片段见paper/settlement_pseudocode.tex。

图片为400 dpi PNG和轮廓化字体的SVG。背景图3200×1920；时间轴3200×1840。

新版论文：research_behavior_draft_v2.tex。已插入背景图、时间轴及结算算法，保留v1。若在已有Overleaf项目中手动插入算法，导言区须加入：
\usepackage{algorithm,algpseudocode}
\floatname{algorithm}{算法}

algorithm与algpseudocode属于排版依赖；使用XeLaTeX。新版图片相对路径为figures/monochrome_revision/，上传时须保持目录结构。本地LaTeX编译服务仍存在沙箱初始化故障，尚未验证新版完整PDF；图片已渲染检查，算法与环境源码核对。

新增算法／两张图片会改变页数和浮动位置，当前未强制分页或锁定页顶布局。

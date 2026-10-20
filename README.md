# 今天去不去工位？基于强化学习的科研行为规划

我们把需求放到一个具体场景中讨论：研究生如何安排科研工作。任务有截止日期，实验室需要有人，睡眠、休息与运动也需要时间。当进度落后时，延长工作时间是一种容易采取的调整，但它是否降低了完成任务的总代价，需要进一步比较。今天去不去工位，看似是地点选择，实际还涉及把有限的时间与状态分配给什么。

上述困境转化为一个可计算、可比较的问题：

> **任务必须完成时，能否减少以透支为代价的完成方式？**

为此，实验固定48单位任务量和四周目标期限，最多允许延期一周，同时计入延期造成的心情损失。研究者每天选择在实验室工作、宿舍休息或健身，行动会影响工作进度、精力、心情、焦虑与导师满意度。我们比较不同固定作息和示范初始化加在线 Q-learning，分别评价最终完成率、按时完成率及整个过程的累积透支成本。

这样的设计回应的是“如何在交付约束下为恢复留出空间”这一部分需求。模型没有覆盖饮食、真实睡眠质量、收入或职场权力关系；它提供一个用于讨论安排与代价的仿真实验，尚不能作为经过现实验证的健康管理或劳动问题解决方案。

## 我们希望提供什么

本项目以严谨的手段进行娱乐：把熟悉的科研生活困境写成明确规则，让一张作息表和一个学习算法接受相同评价，并公开参数、源码和结果供他人检查与修改。AI参与辅助建模、代码检查、实验整理与写作，结论则依据实际记录的结果，不因使用了 AI 而预设算法更优。

我们希望提供的，是一种同时检查交付要求和个人代价的方法，以及继续调研、实验与改进的起点。任务需要完成，恢复也应进入计划；项目需要耗材，研究者不应是耗材。

## 仓库内容与材料状态

本目录整理最终参数下的源码、实验数据、论文和配图，作为 GitHub 仓库内容。作者：陌玉菜菜子（第一作者、通讯作者）、小智（第二作者）。共同地址：东大村。通讯邮箱：`mon4ter@mail.nwpu.edu.cn`。



## 目录

```text
research_rl_handoff/                 完整 Python 源文件及环境测试
  time_experiment.py                最终实验入口：搜索、训练、评价、敏感性回放
  results_time_aware_v3/             最终结果、逐回合数据、日程与三个 Q 表
  results_grace/fixed_ontime_plan.json    历史对照计划（入口运行依赖）
  results_time_aware/selected_plan.json   历史对照计划（入口运行依赖）
paper/
  research_behavior_draft_v3.tex     当前本地论文版本
  LOGO1.png、LOGO2.png               排版资源
  figures/                          最终选用的配图系列、绘图脚本和数据
    monochrome_revision/            黑白时间轴与背景简笔画
    group2_schedules_blue/          六种策略的蓝色作息热图
    group3_progress_multicolor/     多色进度对比
    group4_state_changes/           疲劳、心情、焦虑、导师满意度
  README.md                         论文编译与待补文件说明
docs/TIME_ANALYSIS_V3.md             最终参数订正、方法及结果说明
template_reference/                 用户提供的原始中文 LaTeX 模板参考
verify_release.py                   参数、结果与代表性轨迹验证
MANIFEST.sha256                     整理时文件校验清单
requirements.txt                    Python 依赖
.gitignore                         本地环境与 LaTeX 中间文件排除规则
```

完整 Python 源码包含历史探索脚本，但论文结果以 `time_experiment.py` 和 `results_time_aware_v3` 为准。历史实验的全部输出、旧论文、缓存、第三方安装目录与重复压缩包未收录；部分历史脚本依赖未收录的旧结果，不属于本 README 的最终复现流程。

## 环境与规则

建议 Python 3.10 或以上（代码使用字典合并操作），绘图依赖 Matplotlib 和 NumPy。工作量为48单位，目标28天，最多延期7天；每个延期日上午额外扣除2点心情。每天有上午、下午、晚上三个决策时段。

| 动作 | 上午／下午／晚上工作量 | 精力变化 | 心情变化 | 焦虑变化 |
|---|---|---|---|---|
| 实验室 | 1 / 1.1 / 0.9 | −7 | −4 | +1 |
| 宿舍 | 0 | +8 / +6 / +12 | +1 / 0 / +3 | −6 |
| 健身房 | 0 | −2 | +7 | −8 |

自然恢复仅在晚间结束及相关评价后结算。具体实现见 `environment.py`；参数见 `config.py` 和最终结果的 `settings.json`。源码保留 legacy 配置用于历史兼容；最终入口显式选择 `time_aware` 与详细观测，不能仅看全局默认值判断最终实验参数。

## 安装与检查

在本目录中运行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS：source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s research_rl_handoff -p "test_*.py" -v
python verify_release.py
```

验证脚本检查最终参数、六种策略的统计结果，以及同一环境下固定策略和三个 Q 表的代表性轨迹。它不重新训练，也不证明其他环境下的表现。

## 回放已保存策略

```bash
cd research_rl_handoff
python time_experiment.py --outdir results_time_aware_v3 --replay fixed --seed 4000000
python time_experiment.py --outdir results_time_aware_v3 --replay simple --seed 4000000
python time_experiment.py --outdir results_time_aware_v3 --replay rl --seed 4000000
```

`fixed` 为优化固定日程，`simple` 为简单三天循环，`rl` 加载训练种子42的 Q 表。以上回放不修改已保存结果。

## 完整搜索与训练

在 `research_rl_handoff` 中运行：

```bash
python time_experiment.py --outdir results_reproduced
```

该命令重新筛选3、6、9时段的循环、局部优化固定日程，并对训练种子42、43、44分别执行2000回合示范初始化和10000回合在线 Q-learning。输出到新目录，保留发布时的结果。最终评价使用100个共同环境种子4000000至4000099；RL为三个训练种子各评价100回合，共300次运行，仅100种独立环境安排。敏感性比较重放同一延期日程，不针对每日心情扣除1、2、4重新搜索或训练。

候选排列中使用 Python 集合，跨版本的遍历顺序及同分候选选择可能影响重新搜索的路径；本仓库保存原始最终日程和 Q 表，便于直接回放核查。完整重训可能耗时较长。

## 主要结果

| 策略 | 完成率 | 按时率 | 平均透支成本 | 完成时点 | 最终心情 | 导师满意度 |
|---|---:|---:|---:|---|---:|---:|
| 常规固定作息 | 100% | 100% | 49.2100 | 第23天下午 | 0 | 99.18 |
| 上午补觉 | 100% | 100% | 58.4558 | 第24天晚上 | 0 | 65.60 |
| 简单三天循环 | 100% | 100% | 6.2638 | 第28天上午 | 56 | 95.22 |
| 优化固定日程 | 100% | 100% | 6.1838 | 第28天下午 | 54 | 93.32 |
| 示范初始化加在线 Q-learning | 100% | 100% | 6.1838 | 第28天下午 | 54 | 93.32 |
| 延期低成本循环 | 100% | 0% | 5.5051 | 第35天下午 | 58 | 3.62 |

这些比例描述有限评价样本。RL复现示范表现，未显示优于优化固定日程，也不能解释为从零发现最优作息。降低综合成本不代表所有分项同时改善。精力、心情和焦虑均为未经现实数据校准的模拟变量，结果不构成健康建议。项目可以结题，研究者不应成为耗材。

## 配图与论文

生成配图的命令在仓库根目录运行：

```bash
python paper/figures/monochrome_revision/generate_revision.py
python paper/figures/group2_schedules_blue/generate_heatmaps_blue.py
python paper/figures/group3_progress_multicolor/generate_progress_multicolor.py
python paper/figures/group4_state_changes/generate_state_changes.py
```

当前绘图脚本沿用 Windows 字体路径 `C:/Windows/Fonts/msyh.ttc`。其他系统需要安装可用中文字体，并调整脚本中的 `FONT` 或 `font_path` 后运行。已生成 PNG 和 SVG 可直接查看；SVG文字已转为路径。绘图脚本会覆盖各自目录的配图及数据，请在需要重新生成时使用。


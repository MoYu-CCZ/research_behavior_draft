"""Slot-level MDP. Deadline is a target; episodes end at full completion or failure."""
import random
from config import *

class ResearchEnv:
    def __init__(self, weeks=DEADLINE_WEEKS, seed=0, total_work=None,
                 grace_days=7, delay_mood_penalty=2, reward_mode='legacy', state_mode='compact', model_profile='legacy'):
        if not isinstance(weeks, int) or weeks < 1:
            raise ValueError('weeks must be a positive integer')
        self.weeks = weeks
        self.deadline = weeks * 21
        self.total_work = weeks * 12 if total_work is None else total_work
        if not isinstance(self.total_work, int) or self.total_work < 1:
            raise ValueError('total_work must be a positive integer')
        if not isinstance(grace_days, int) or not 0 <= grace_days <= 7 or delay_mood_penalty < 0:
            raise ValueError('Invalid grace days or delay mood penalty')
        if reward_mode not in ('legacy', 'cost'):
            raise ValueError('Unknown reward mode')
        self.max_steps = self.deadline + grace_days * 3
        self.delay_mood_penalty = delay_mood_penalty
        self.reward_mode = reward_mode
        if state_mode not in ('compact', 'detailed'):
            raise ValueError('Unknown state mode')
        self.state_mode = state_mode
        if model_profile not in ('legacy', 'time_aware'):
            raise ValueError('Unknown model profile')
        self.model_profile = model_profile
        self.rng = random.Random(seed)

    @property
    def work(self):
        return self._work_ticks / 10

    @work.setter
    def work(self, value):
        self._work_ticks = round(value * 10)

    def reset(self):
        self.t = 0
        self.energy, self.mood, self.mentor, self.anxiety = 100, 60, 60, 0
        self.work = self.total_work
        self.done = False
        self.logs, self.slot_logs, self.day_actions = [], [], []
        self.checks = {}
        for week in range((self.max_steps + 20) // 21):
            days = self.rng.sample(range(5), self.rng.randint(0, MAX_WEEKLY_CHECKS))
            self.checks[week] = {(d, self.rng.randrange(2)) for d in days}
        self.cost = dict(fatigue=0.0, anxiety=0.0, low_mood=0.0, low_energy=0.0)
        self.return_sum = 0.0
        self.reason = ''
        return self.observation()

    def observation(self):
        if self.state_mode == 'detailed':
            return (self.t, self._work_ticks if self.model_profile == 'time_aware' else self.work, max(0, int(self.energy)) // 10,
                    int(self.mood) // 5, int(self.mentor) // 20,
                    int(self.anxiety) // 10, self.day_actions.count('实验室'))
        # Compact time/progress aggregation; future checks are never exposed.
        remaining = max(1, self.deadline - self.t)
        pace = min(4, int(4 * self.work / remaining))
        return (self.t % 3, (self.t // 3) % 7 < 5,
                min(4, self.t * 4 // self.deadline), pace,
                min(4, max(0, int(self.energy)) // 25),
                min(4, int(self.mood) // 25), min(4, int(self.mentor) // 25),
                min(4, int(self.anxiety) // 25), self.work <= 3, self.day_actions.count('实验室'))

    def allowed_actions(self):
        # One-step energy guard; used consistently in training and evaluation.
        return [i for i, a in enumerate(ACTIONS)
                if self.energy + ACTION_EFFECTS[a]['energy']
                - (FULL_LAB_DAY_EXTRA_ENERGY_LOSS
                   if self.t % 3 == 2 and self.day_actions == ['实验室'] * 2
                   and a == '实验室' else 0) >= 0]

    def step(self, action):
        if self.done:
            raise RuntimeError('step after episode end')
        name = ACTIONS[action]
        effect = ACTION_EFFECTS[name].copy()
        if self.model_profile == 'time_aware':
            if name == '实验室':
                effect['work'] = LAB_WORK_BY_SLOT[self.t % 3]
            elif name == '宿舍':
                effect['energy'] = DORM_ENERGY_BY_SLOT[self.t % 3]
                effect['mood'] = DORM_MOOD_BY_SLOT[self.t % 3]
        before_anxiety = self.anxiety
        day, slot = self.t // 3, self.t % 3
        delay_mood_loss = self.delay_mood_penalty if self.t >= self.deadline and slot == 0 else 0
        self.mood = clamp(self.mood - delay_mood_loss)
        self.energy = min(100, self.energy + effect['energy'])
        self.mood = clamp(self.mood + effect['mood'])
        self.anxiety = clamp(self.anxiety + effect['anxiety'])
        progress = min(self.work, effect['work'])
        self.work = self.work - progress
        day, slot = self.t // 3, self.t % 3
        checked = ((day % 7, slot) in self.checks[day // 7])
        if checked:
            self.mentor = clamp(self.mentor + (6 if name == '实验室' else -12))
        self.day_actions.append(name)
        self.t += 1
        failed = self.energy < 0 or self.anxiety >= 100
        daily_gain = weekly_gain = 0
        if slot == 2 and not failed:
            if self.day_actions == ['实验室'] * 3:
                self.energy -= FULL_LAB_DAY_EXTRA_ENERGY_LOSS
                self.mood = clamp(self.mood - FULL_LAB_DAY_EXTRA_MOOD_LOSS)
                self.anxiety = clamp(self.anxiety + FULL_LAB_DAY_EXTRA_ANXIETY_GAIN)
            # Check exhaustion BEFORE recovery or clipping.
            failed = self.energy < 0 or self.anxiety >= 100
            if not failed:
                daily_gain = sum(self.mood < x for x in (50, 35, 20, 10))
                daily_gain += sum(self.mentor < x for x in (50, 35, 20, 10))
                lag = max(0, self.total_work * min(self.t / self.deadline, 1)
                          - (self.total_work - self.work))
                daily_gain += int(lag * 0.15)
                self.anxiety = clamp(self.anxiety + daily_gain)
                if (day + 1) % 7 == 0:
                    expected = self.total_work * min(self.t / self.deadline, 1)
                    lag = max(0, expected - (self.total_work - self.work))
                    self.mentor = clamp(self.mentor + (8 if lag == 0 else -14))
                    weekly_gain = int(2 + min(day // 7 + 1, self.weeks) + 0.3 * lag) if lag else 0
                    self.anxiety = clamp(self.anxiety + weekly_gain)
                failed = self.anxiety >= 100
                if not failed:
                    recovery = max(2, BASE_DAILY_ENERGY_RECOVERY - int(self.anxiety / 20))
                    if self.model_profile == 'time_aware' and name == '实验室':
                        recovery = max(2, recovery - EVENING_LAB_RECOVERY_PENALTY)
                    self.energy = clamp(self.energy + recovery)
        costs = dict(fatigue=(100 - max(0, self.energy)) / 100,
                     anxiety=self.anxiety / 100,
                     low_mood=max(0, 30 - self.mood) / 30,
                     low_energy=max(0, 25 - self.energy) / 25)
        for k, v in costs.items():
            self.cost[k] += v
        reward = 20 * progress - sum(costs.values())
        if self.reward_mode == 'legacy' and self.t > self.deadline:
            reward -= 8
        success = self.work == 0 and not failed
        terminated = failed or success
        truncated = self.t >= self.max_steps and not terminated
        if failed:
            reward -= 500
            self.reason = '精力耗尽' if self.energy < 0 else '焦虑达到100'
        elif success:
            reward += (400 if self.t <= self.deadline else 150) if self.reward_mode == 'legacy' else 1000
            self.reason = '按时完成' if self.t <= self.deadline else '延期完成'
        elif truncated:
            reward -= 500
            self.reason = '达到延期上限，未完成'
        elif self.reward_mode == 'legacy' and self.t == self.deadline and self.work:
            reward -= 150
        self.done = terminated or truncated
        self.return_sum += reward
        self.slot_logs.append(dict(step=self.t, action=name, checked=checked,
                                   anxiety_delta=self.anxiety - before_anxiety, delay_mood_loss=delay_mood_loss,
                                   work_progress=progress, energy=self.energy, mood=self.mood, **costs))
        if slot == 2 or self.done:
            self.logs.append(dict(day=day + 1, actions=list(self.day_actions),
                                  energy=self.energy, mood=self.mood, mentor=self.mentor,
                                  anxiety=self.anxiety, work_remaining=self.work,
                                  daily_anxiety_gain=daily_gain, weekly_anxiety_gain=weekly_gain))
        if slot == 2:
            self.day_actions = []
        return self.observation(), reward, terminated, truncated, dict(success=success)

    def result(self):
        success = self.work == 0 and self.reason in ('按时完成', '延期完成')
        return dict(success=success, on_time=success and self.t <= self.deadline,
                    completion=1 - self.work / self.total_work, reason=self.reason,
                    steps=self.t, overtime_steps=max(0, self.t - self.deadline),
                    total_work=self.total_work, deadline_steps=self.deadline, model_profile=self.model_profile,
                    logs=self.logs, slot_logs=self.slot_logs, costs=self.cost.copy(),
                    total_cost=sum(self.cost.values()), return_sum=self.return_sum,
                    delay_days=(max(0, self.t - self.deadline) + 2) // 3,
                    final_energy=self.energy, final_mood=self.mood,
                    final_mentor=self.mentor, final_anxiety=self.anxiety,
                    final_work_remaining=self.work)

def clamp(x):
    return max(0, min(100, x))

def simulate_one_episode(agent=None, train=False, seed=0, weeks=4, policy='rl', total_work=None, **env_options):
    env = ResearchEnv(weeks, seed, total_work=total_work, **env_options)
    obs = env.reset()
    decision_rng = agent.rng if train and agent is not None else random.Random(seed ^ 0x13579)
    while not env.done:
        allowed = env.allowed_actions()
        if policy == 'rl':
            a = agent.choose_action(obs, explore=train, allowed=allowed, rng=decision_rng)
        elif policy == 'random':
            a = env.rng.choice(allowed)
        elif policy == 'fixed':
            a = 0 if env.t % 3 < 2 else 1
        elif policy == 'lab':
            a = 0
        else:
            raise ValueError(policy)
        if a not in allowed:
            a = 1
        next_obs, reward, terminated, truncated, _ = env.step(a)
        if train:
            agent.learn(obs, a, reward, next_obs, terminated or truncated,
                        env.allowed_actions())
        obs = next_obs
    return env.result()

def train_agent(agent, episodes, decay_interval=700, decay_rate=0.9, min_epsilon=0.02,
                seed=42, weeks=4, total_work=None, **env_options):
    history = []
    for ep in range(episodes):
        r = simulate_one_episode(agent, True, seed + ep, weeks, total_work=total_work, **env_options)
        history.append((int(r['success']), r['final_work_remaining']))
        if (ep + 1) % decay_interval == 0:
            agent.epsilon = max(min_epsilon, agent.epsilon * decay_rate)
    return history








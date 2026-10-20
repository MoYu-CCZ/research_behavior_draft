import unittest
from agent import QLearningAgent
from config import ACTIONS
from environment import ResearchEnv, simulate_one_episode

class TransitionTests(unittest.TestCase):
    def test_work_and_checks(self):
        for weeks in (1, 2, 4, 8):
            env = ResearchEnv(weeks, 4)
            env.reset()
            self.assertEqual(env.total_work, 12 * weeks)
            self.assertEqual(env.deadline, 21 * weeks)
            for plan in env.checks.values():
                self.assertLessEqual(len(plan), 3)
                self.assertEqual(len({d for d, _ in plan}), len(plan))
                self.assertTrue(all(0 <= d < 5 and s in (0, 1) for d, s in plan))

    def test_day_transition_and_week_transition(self):
        env = ResearchEnv(1, 0)
        env.reset()
        for _ in range(21):
            obs, _, _, _, _ = env.step(1)
        self.assertEqual(obs, env.observation())
        self.assertEqual(env.t, 21)
        self.assertEqual(env.day_actions, [])
        self.assertEqual(env.logs[-1]['mentor'], env.mentor)
        self.assertEqual(env.logs[-1]['anxiety'], env.anxiety)
        self.assertFalse(env.done)
        self.assertEqual(env.work, 12)

    def test_exhaustion_before_recovery(self):
        env = ResearchEnv(1, 0)
        env.reset()
        env.t = 2
        env.day_actions = ['实验室', '实验室']
        env.energy = 10
        _, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(env.energy, -1)

    def test_completion_terminal_reward_and_guard(self):
        env = ResearchEnv(1, 0)
        env.reset()
        env.work = 1
        _, reward, terminal, truncated, _ = env.step(0)
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertGreater(reward, 400)
        self.assertEqual(env.result()['completion'], 1)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_overtime_and_limit(self):
        env = ResearchEnv(1, 0)
        env.reset()
        env.t, env.work = 21, 1
        env.step(0)
        self.assertTrue(env.result()['success'])
        self.assertFalse(env.result()['on_time'])
        self.assertEqual(env.result()['overtime_steps'], 1)
        env.reset()
        env.t = env.max_steps - 1
        _, _, terminated, truncated, _ = env.step(1)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(env.result()['success'])

    def test_anxiety_failure(self):
        env = ResearchEnv(1, 0)
        env.reset()
        env.anxiety = 99
        _, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated)

    def test_evaluation_no_exploration_or_learning(self):
        agent = QLearningAgent(ACTIONS, epsilon=1)
        original = agent.choose_action
        def checked(key, explore=True, allowed=None, rng=None):
            self.assertFalse(explore)
            return original(key, explore, allowed, rng)
        agent.choose_action = checked
        simulate_one_episode(agent, train=False, seed=5, weeks=1)
        self.assertEqual(agent.q_table, {})

    def test_cost_sums(self):
        result = simulate_one_episode(policy='fixed', weeks=1, seed=9)
        self.assertAlmostEqual(result['total_cost'], sum(result['costs'].values()))
        self.assertEqual(len(result['slot_logs']), result['steps'])
        self.assertEqual(result['completion'], 1)

    def test_six_day_full_lab_calibration(self):
        env = ResearchEnv(4, 42)
        env.reset()
        while not env.done and env.t < 18:
            env.step(0)
        self.assertGreaterEqual(env.t, 15)
        self.assertLessEqual(env.t, 18)
        self.assertLessEqual(env.energy, 10)

    def test_terminal_q_update_has_no_bootstrap(self):
        agent = QLearningAgent(ACTIONS, alpha=1)
        agent.q_table['next'] = [10000, 10000, 10000]
        agent.learn('state', 0, -500, 'next', done=True)
        self.assertEqual(agent.q_table['state'][0], -500)

    def test_energy_cap_and_common_guard(self):
        env = ResearchEnv(1, 0)
        env.reset()
        env.step(1)
        self.assertEqual(env.energy, 100)
        env.energy = 3
        self.assertNotIn(0, env.allowed_actions())
        self.assertIn(1, env.allowed_actions())

    def test_extended_deadline_preserves_work(self):
        env = ResearchEnv(5, 42, total_work=48)
        env.reset()
        self.assertEqual(env.deadline, 105)
        self.assertEqual(env.work, 48)
        self.assertEqual(ResearchEnv(4, 42).total_work, 48)
        env.t, env.work = 84, 1
        env.step(0)
        self.assertTrue(env.result()['on_time'])
        self.assertEqual(env.result()['completion'], 1)
        with self.assertRaises(ValueError):
            ResearchEnv(5, 42, total_work=0)

    def test_target_and_seven_day_grace(self):
        env = ResearchEnv(4, 1)
        env.reset()
        self.assertEqual(env.deadline, 84)
        self.assertEqual(env.max_steps, 105)
        env.t, env.work = 84, 1
        env.step(0)
        self.assertTrue(env.result()['success'])
        self.assertFalse(env.result()['on_time'])
        self.assertEqual(env.result()['delay_days'], 1)
        self.assertEqual(env.mood, 54)
        self.assertEqual(env.slot_logs[-1]['delay_mood_loss'], 2)

    def test_delay_mood_once_each_day(self):
        env = ResearchEnv(4, 1)
        env.reset()
        env.t = 84
        for _ in range(3):
            env.step(1)
        self.assertEqual(sum(r['delay_mood_loss'] for r in env.slot_logs), 2)
        env.step(1)
        self.assertEqual(sum(r['delay_mood_loss'] for r in env.slot_logs), 4)

    def test_evaluation_preserves_training_rng(self):
        agent = QLearningAgent(ACTIONS)
        before = agent.rng.getstate()
        simulate_one_episode(agent, seed=8, weeks=1)
        self.assertEqual(agent.rng.getstate(), before)

    def test_undiscounted_cost_objective(self):
        env = ResearchEnv(4, 2, reward_mode='cost')
        env.reset()
        env.t, env.work = 84, 1
        _, reward, _, _, _ = env.step(0)
        self.assertAlmostEqual(reward, 1020 - env.result()['total_cost'])

    def test_detailed_state_distinguishes_cost_threshold(self):
        env = ResearchEnv(4, 1, state_mode='detailed')
        env.reset()
        env.mood = 29
        low = env.observation()
        env.mood = 30
        recovered = env.observation()
        self.assertNotEqual(low, recovered)

    def test_no_extension_beyond_one_week(self):
        with self.assertRaises(ValueError):
            ResearchEnv(4, 1, grace_days=8)

    def test_time_aware_slot_work_and_exact_completion(self):
        env = ResearchEnv(4, 1, model_profile='time_aware')
        env.reset()
        for slot, expected in enumerate((1, 1.1, 0.9)):
            env.t, env.work, env.energy, env.day_actions = slot, 10, 100, []
            env.step(0)
            self.assertAlmostEqual(env.work, 10 - expected)
        env.reset()
        env.t, env.work = 1, 1.1
        env.step(0)
        self.assertEqual(env.work, 0)
        self.assertTrue(env.result()['success'])
        self.assertEqual(env._work_ticks, 0)

    def test_rest_by_slot_and_night_recovery_timing(self):
        env = ResearchEnv(4, 1, model_profile='time_aware')
        env.reset()
        env.energy = 50
        env.step(1)
        self.assertEqual(env.energy, 58)
        self.assertEqual(env.mood, 61)
        env.step(1)
        self.assertEqual(env.energy, 64)
        self.assertEqual(env.mood, 61)
        env.step(1)
        self.assertEqual(env.energy, 84)
        self.assertEqual(env.mood, 64)

    def test_evening_lab_reduces_subsequent_recovery(self):
        env = ResearchEnv(4, 1, model_profile='time_aware')
        env.reset()
        env.t, env.energy = 2, 50
        env.step(0)
        self.assertEqual(env.energy, 49)

if __name__ == '__main__':
    unittest.main()









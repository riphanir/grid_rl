"""
train.py
--------
نقطة الدخول الرئيسية للمشروع:
1) يحمّل النموذج المدرَّب سابقاً إن وُجد (trained_agent.json) بدل
   البدء من الصفر -> بهذا "لا ينسى" الوكيل ما تعلمه في أي تشغيلة
   سابقة (محلياً أو عبر GitHub Actions).
2) يدرّب الوكيل عبر مراحل عشوائية متتالية: أول ما يحل مرحلة
   تتولّد مرحلة جديدة تلقائياً.
3) يحفظ النموذج المحدَّث + إحصائيات تراكمية (عدد المراحل
   المحلولة إجمالاً منذ أول تشغيلة) في نهاية كل تشغيلة.
4) اختيارياً يعرض الوكيل وهو يحل بضع مراحل جديدة في الطرفية.

تشغيل محلي بسيط:
    python3 train.py

تشغيل مخصص (مثال لما يُستخدم من GitHub Actions):
    python3 train.py --episodes 3000 --no-demo
"""

import argparse
import time
import random

from world import GridWorld, ACTIONS
from agent import QLearningAgent
from level_generator import generate_level
from progress import load_stats, save_stats


# إعدادات عامة للمراحل
WIDTH, HEIGHT = 10, 8
WALL_DENSITY = 0.15
NUM_BOXES = 1
MAX_STEPS = 150

MODEL_PATH = "trained_agent.json"
STATS_PATH = "training_stats.json"


def make_env(rng):
    """يبني مرحلة عشوائية جديدة (عالم GridWorld)."""
    walls, boxes, goal, start = generate_level(
        WIDTH, HEIGHT, WALL_DENSITY, NUM_BOXES, rng=rng
    )
    return GridWorld(
        width=WIDTH, height=HEIGHT,
        walls=walls, boxes=boxes,
        goal=goal, agent_start=start,
        max_steps=MAX_STEPS,
    )


def load_or_create_agent(model_path, seed=None):
    """
    يحمّل نموذجاً مدرَّباً سابقاً (جدول Q + epsilon) إن وُجد،
    وإلا ينشئ وكيلاً جديداً من الصفر. هذا هو أساس "عدم النسيان"
    بين تشغيلة وأخرى.
    """
    agent = QLearningAgent(actions=ACTIONS, seed=seed)
    try:
        agent.load(model_path)
        print(f"🔄 تم تحميل نموذج سابق من '{model_path}' "
              f"({len(agent.q)} قيمة محفوظة، epsilon={agent.epsilon:.3f}) "
              f"-- متابعة التعلّم بدل البدء من الصفر.")
    except FileNotFoundError:
        print(f"🆕 لا يوجد نموذج سابق في '{model_path}' -- بدء تعلّم جديد.")
    return agent


def train(agent, total_episodes=3000, log_every=300, seed=None,
          verbose=True, start_level_number=1):
    """
    يدرّب الوكيل عبر مراحل متتالية:
    - يلعب حلقة (episode) على المرحلة الحالية.
    - إذا وصل للهدف -> يولّد مرحلة جديدة عشوائياً للحلقة التالية.
    - إذا انتهى الوقت بدون وصول -> يعيد المحاولة على *نفس* المرحلة.
    """
    rng = random.Random(seed)
    env = make_env(rng)

    rewards_history = []
    level_number = start_level_number
    solved_count = 0
    episodes_on_current_level = 0

    for ep in range(1, total_episodes + 1):
        env.reset()
        state = env.state_key()
        total_reward = 0.0
        done = False
        obs = None
        episodes_on_current_level += 1

        while not done:
            action = agent.choose_action(state)
            obs, reward, done, _info = env.step(action)
            next_state = env.state_key()
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if obs["on_goal"]:
            solved_count += 1
            if verbose:
                print(f"✅ تم حل المرحلة رقم {level_number} "
                      f"(بعد {episodes_on_current_level} محاولة على هذه المرحلة، "
                      f"محلياً في هذه التشغيلة: {ep}) -- مرحلة جديدة عشوائياً...")

            walls, boxes, goal, start = generate_level(
                WIDTH, HEIGHT, WALL_DENSITY, NUM_BOXES, rng=rng
            )
            env.new_level(walls, boxes, goal, start)
            level_number += 1
            episodes_on_current_level = 0

        if verbose and ep % log_every == 0:
            avg = sum(rewards_history[-log_every:]) / log_every
            print(f"— الإحصائيات: المحاولة {ep:5d}/{total_episodes} | متوسط "
                  f"المكافأة آخر {log_every}: {avg:7.2f} | مراحل تم حلّها "
                  f"بهذه التشغيلة: {solved_count} | epsilon={agent.epsilon:.3f}")

    return env, agent, rewards_history, solved_count, level_number


def demo(agent, num_levels=3, delay=0.2, max_steps=150, seed=None):
    """يعرض الوكيل المدرَّب وهو يحل عدة مراحل جديدة بالتتابع."""
    rng = random.Random(seed)
    env = make_env(rng)

    for level in range(1, num_levels + 1):
        env.reset()
        state = env.state_key()
        print(f"\n================ المرحلة التجريبية {level} ================\n")
        print(env.render())
        print()

        done = False
        solved_this_level = False
        for step in range(max_steps):
            action = agent.choose_action(state, greedy=True)  # بدون عشوائية
            obs, reward, done, info = env.step(action)
            state = env.state_key()

            print(f"الخطوة {step + 1}: فعل={action:<6} مكافأة={reward:6.1f} "
                  f"| الحدث: {info['event']}")
            print(env.render())
            print()
            if delay:
                time.sleep(delay)

            if done:
                if obs["on_goal"]:
                    print(f"🎉 حل الوكيل المرحلة {level} في {step + 1} خطوة!")
                    solved_this_level = True
                else:
                    print(f"⏱️ فشل الوكيل في حل المرحلة {level} (انتهى الوقت).")
                break

        if not solved_this_level and not done:
            print(f"⏱️ لم يكمل الوكيل المرحلة {level} خلال {max_steps} خطوة "
                  f"المخصصة للعرض.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="تدريب وكيل Q-Learning على مراحل شبكية عشوائية متتالية."
    )
    parser.add_argument("--episodes", type=int, default=3000,
                         help="عدد محاولات التدريب في هذه التشغيلة (افتراضي: 3000)")
    parser.add_argument("--log-every", type=int, default=300,
                         help="طباعة إحصائيات كل كم محاولة (افتراضي: 300)")
    parser.add_argument("--seed", type=int, default=None,
                         help="بذرة عشوائية لتوليد المراحل (افتراضي: عشوائية كل تشغيلة)")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                         help="مسار ملف النموذج المحفوظ")
    parser.add_argument("--stats-path", type=str, default=STATS_PATH,
                         help="مسار ملف الإحصائيات التراكمية")
    parser.add_argument("--no-demo", action="store_true",
                         help="تخطي عرض الوكيل في نهاية التشغيلة (مفيد لـ CI)")
    parser.add_argument("--demo-levels", type=int, default=3,
                         help="عدد المراحل التي تُعرض بعد التدريب")
    parser.add_argument("--delay", type=float, default=0.15,
                         help="التأخير بالثواني بين كل خطوة في العرض (0 = بدون تأخير)")
    parser.add_argument("--quiet", action="store_true",
                         help="تقليل الطباعة أثناء التدريب")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    stats = load_stats(args.stats_path)
    stats["total_runs"] += 1
    print(f"===== تشغيلة رقم {stats['total_runs']} =====")
    print(f"إجمالي المراحل المحلولة سابقاً (كل التشغيلات): "
          f"{stats['total_levels_solved']}")
    print(f"إجمالي محاولات التدريب سابقاً: {stats['total_episodes']}\n")

    agent = load_or_create_agent(args.model_path, seed=args.seed)

    env, agent, history, solved_count, next_level_number = train(
        agent,
        total_episodes=args.episodes,
        log_every=args.log_every,
        seed=args.seed,
        verbose=not args.quiet,
        start_level_number=stats["last_level_number"],
    )

    # تحديث الإحصائيات التراكمية وحفظها
    stats["total_episodes"] += args.episodes
    stats["total_levels_solved"] += solved_count
    stats["last_level_number"] = next_level_number
    save_stats(args.stats_path, stats)

    agent.save(args.model_path)

    print(f"\n=== انتهت هذه التشغيلة: تم حل {solved_count} مرحلة جديدة ===")
    print(f"=== الإجمالي التراكمي منذ أول تشغيلة: "
          f"{stats['total_levels_solved']} مرحلة عبر {stats['total_runs']} تشغيلة ===")
    print(f"تم حفظ النموذج المحدَّث في '{args.model_path}'")
    print(f"تم حفظ الإحصائيات في '{args.stats_path}'\n")

    if not args.no_demo:
        demo(agent, num_levels=args.demo_levels, delay=args.delay, seed=None)

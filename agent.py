"""
agent.py
--------
الوكيل (Agent): "نموذج" قرار بسيط يتعلم بخوارزمية Q-Learning الجدولية
(Tabular Q-Learning). هذا أبسط شكل من "التعلّم" ويعطي نتائج جيدة
جداً في عوالم صغيرة مثل عالمنا، بدون الحاجة لشبكات عصبية أو
بيانات ضخمة.

الفكرة:
- لكل (حالة، فعل) نحتفظ بقيمة Q تمثّل "مدى جودة" هذا الفعل.
- الوكيل يجرّب أفعالاً عشوائية أحياناً (استكشاف) وأفعالاً جيدة
  معروفة أحياناً أخرى (استغلال) -> epsilon-greedy.
- بعد كل خطوة، نحدّث القيمة باستخدام معادلة بيلمان (Bellman).
"""

import json
import random


class QLearningAgent:
    def __init__(self, actions, alpha=0.15, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 seed=None):
        self.actions = actions
        self.alpha = alpha          # معدل التعلّم
        self.gamma = gamma          # أهمية المستقبل
        self.epsilon = epsilon      # نسبة الاستكشاف العشوائي
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q = {}  # (state, action) -> value

    def _get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, greedy=False):
        """يختار فعلاً: عشوائي (استكشاف) أو الأفضل حالياً (استغلال)."""
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)

        q_values = [self._get_q(state, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return self.rng.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """تحديث Q-value باستخدام معادلة بيلمان."""
        old_value = self._get_q(state, action)
        if done:
            target = reward
        else:
            next_max = max(self._get_q(next_state, a) for a in self.actions)
            target = reward + self.gamma * next_max

        new_value = old_value + self.alpha * (target - old_value)
        self.q[(state, action)] = new_value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------------------------------------------------------
    # حفظ / تحميل النموذج المدرَّب
    # ---------------------------------------------------------
    def save(self, path):
        """
        يحفظ النموذج بالكامل: جدول Q + نسبة الاستكشاف الحالية
        (epsilon) + بقية المعاملات. حفظ epsilon مهم جداً حتى لا
        "ينسى" الوكيل تقدّمه ويعود للاستكشاف العشوائي بالكامل
        في كل تشغيلة جديدة (مثلاً عند التشغيل عبر GitHub Actions).
        """
        data = {
            "actions": self.actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q": {repr(k): v for k, v in self.q.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.actions = data["actions"]
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_min = data.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.q = {}
        for k_str, v in data["q"].items():
            key = eval(k_str)  # آمن هنا لأننا كتبنا الملف بأنفسنا
            self.q[key] = v

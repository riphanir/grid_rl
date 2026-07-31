"""
world.py
--------
عالم شبكي 2D بسيط (Grid World) يحتوي على:
- أرض (ground)
- جدران (walls)
- صناديق قابلة للدفع (boxes)
- هدف (goal)
- قوانين فيزيائية بسيطة: حركة، تصادم، دفع الصناديق.

هذا الملف مسؤول فقط عن "العالم" (المرحلة 1 و 2).
"""

import random

# اتجاهات الحركة الممكنة (dx, dy)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ACTIONS = ["up", "down", "left", "right", "wait"]

# رموز الرسم في الطرفية (Terminal)
SYMBOLS = {
    "empty": ".",
    "wall": "#",
    "box": "B",
    "goal": "G",
    "agent": "A",
    "agent_on_goal": "@",
}


class GridWorld:
    """
    عالم شبكي بسيط بقوانين فيزيائية أساسية:
    - الوكيل يتحرك خطوة واحدة في كل مرة.
    - إذا اصطدم بجدار: يبقى في مكانه ويُعاقَب.
    - إذا حاول التحرك نحو صندوق: الصندوق يُدفع خطوة إضافية
      بنفس الاتجاه، إلا إذا كان هناك جدار أو صندوق آخر خلفه
      (وعندها تُمنع الحركة بالكامل - قانون تصادم).
    - الوصول لخلية الهدف ينهي الحلقة (episode) بمكافأة كبيرة.
    """

    def __init__(self, width, height, walls=None, boxes=None,
                 goal=None, agent_start=None, max_steps=200, seed=None):
        self.width = width
        self.height = height
        self.walls = set(walls or [])
        self.initial_boxes = list(boxes or [])
        self.goal = goal
        self.agent_start = agent_start or (0, 0)
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.reset()

    # ---------------------------------------------------------
    # إدارة الحالة (State management)
    # ---------------------------------------------------------
    def reset(self):
        """يعيد ضبط العالم لبداية حلقة جديدة (episode)."""
        self.agent_pos = self.agent_start
        self.boxes = set(self.initial_boxes)
        self.steps = 0
        self.visited = {self.agent_pos}  # لتتبع "الاكتشاف" (+5)
        self.done = False
        return self.get_observation()

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def _cell_type(self, pos):
        """يرجع نوع الخلية: wall / box / goal / empty / out"""
        if not self._in_bounds(pos):
            return "out"
        if pos in self.walls:
            return "wall"
        if pos in self.boxes:
            return "box"
        if pos == self.goal:
            return "goal"
        return "empty"

    # ---------------------------------------------------------
    # المرحلة 2: القوانين الفيزيائية (خطوة زمنية واحدة)
    # ---------------------------------------------------------
    def step(self, action):
        """
        ينفذ خطوة زمنية واحدة:
        1) يحسب الموقع الجديد المقترح.
        2) يطبق قوانين الحركة والتصادم.
        3) يحدّث حالة العالم.
        4) يرجع (observation, reward, done, info)
        """
        assert action in ACTIONS, f"فعل غير معروف: {action}"
        self.steps += 1
        reward = -1.0  # تكلفة كل خطوة
        info = {"event": "moved"}

        if action == "wait" or self.done:
            pass
        else:
            dr, dc = DIRS[action]
            new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            target_type = self._cell_type(new_pos)

            if target_type == "wall" or target_type == "out":
                # اصطدام بجدار / حدود العالم
                reward -= 20.0
                info["event"] = "hit_wall"

            elif target_type == "box":
                # محاولة دفع صندوق
                push_pos = (new_pos[0] + dr, new_pos[1] + dc)
                push_type = self._cell_type(push_pos)
                if push_type in ("wall", "box", "out"):
                    # لا يمكن الدفع -> تصادم
                    reward -= 20.0
                    info["event"] = "blocked_push"
                else:
                    # الدفع ناجح
                    self.boxes.remove(new_pos)
                    self.boxes.add(push_pos)
                    self.agent_pos = new_pos
                    info["event"] = "pushed_box"

            else:
                # خلية فارغة أو هدف -> حركة حرة
                self.agent_pos = new_pos
                info["event"] = "moved"

        # مكافأة الاكتشاف: أول مرة يزور فيها الوكيل خلية
        if self.agent_pos not in self.visited:
            self.visited.add(self.agent_pos)
            reward += 5.0
            info["discovered"] = True

        # هل وصل للهدف؟
        if self.agent_pos == self.goal:
            reward += 100.0
            self.done = True
            info["event"] = "reached_goal"

        # انتهاء الوقت
        if self.steps >= self.max_steps:
            self.done = True

        return self.get_observation(), reward, self.done, info

    # ---------------------------------------------------------
    # المرحلة 3 (جزء الرؤية): ما يراه الوكيل
    # ---------------------------------------------------------
    def get_observation(self):
        """
        يرجع رؤية جزئية للوكيل حول موقعه، مثل:
        - موقعي الحالي
        - محتوى الخلايا الأربع المجاورة (فوق/تحت/يمين/يسار)
        - موقع الهدف والمسافة إليه
        """
        neighbors = {}
        beyond = {}  # الخلية التي تليها (خطوتان للأمام) في كل اتجاه
        for direction, (dr, dc) in DIRS.items():
            npos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            neighbors[direction] = self._cell_type(npos)
            npos2 = (self.agent_pos[0] + 2 * dr, self.agent_pos[1] + 2 * dc)
            beyond[direction] = self._cell_type(npos2)

        gr, gc = self.goal
        ar, ac = self.agent_pos
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal,
            "distance": abs(gr - ar) + abs(gc - ac),
            "neighbors": neighbors,       # مثال: {"up": "wall", "right": "box", ...}
            "beyond": beyond,             # ما يوجد خلف كل خلية مجاورة (يفيد لمعرفة إمكانية الدفع)
            "on_goal": self.agent_pos == self.goal,
            "boxes": tuple(sorted(self.boxes)),
            "steps": self.steps,
        }

    @staticmethod
    def _distance_bucket(d):
        """يحوّل المسافة الدقيقة إلى فئة تقريبية (تقليل عدد الحالات)."""
        if d <= 1:
            return 0
        if d <= 3:
            return 1
        if d <= 6:
            return 2
        if d <= 10:
            return 3
        return 4

    def state_key(self):
        """
        تمثيل *نسبي* وقابل للـ hashing لاستخدامه في جدول Q.

        بدل الاعتماد على الإحداثيات المطلقة (التي تختلف مع كل
        مرحلة جديدة)، نستخدم:
        - اتجاه الهدف بالنسبة للوكيل (فوق/تحت × يمين/يسار).
        - فئة المسافة التقريبية للهدف.
        - محتوى الخلايا المجاورة مباشرة (جدار/صندوق/فارغ/هدف).

        هذا يسمح للوكيل بنقل ما تعلّمه ("لو في جدار قدامي، لف")
        إلى مراحل جديدة يولّدها المولّد العشوائي، بدل أن يبدأ
        التعلّم من الصفر في كل مرحلة.
        """
        obs = self.get_observation()
        ar, ac = obs["agent_pos"]
        gr, gc = obs["goal_pos"]
        dr, dc = gr - ar, gc - ac
        dir_r = (dr > 0) - (dr < 0)   # -1 / 0 / 1
        dir_c = (dc > 0) - (dc < 0)   # -1 / 0 / 1

        return (
            dir_r,
            dir_c,
            self._distance_bucket(obs["distance"]),
            tuple(sorted(obs["neighbors"].items())),
            tuple(sorted(obs["beyond"].items())),
        )

    # ---------------------------------------------------------
    # تحميل مرحلة جديدة (لدعم التوليد العشوائي المتتالي)
    # ---------------------------------------------------------
    def new_level(self, walls, boxes, goal, agent_start, width=None, height=None):
        """
        يستبدل تخطيط العالم الحالي (المرحلة) بتخطيط جديد كلياً،
        ويعيد ضبط حالة الوكيل. تُستخدم هذه الدالة بعد حل كل
        مرحلة لتحميل مرحلة عشوائية جديدة.
        """
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        self.walls = set(walls)
        self.initial_boxes = list(boxes)
        self.goal = goal
        self.agent_start = agent_start
        return self.reset()

    # ---------------------------------------------------------
    # الرسم في الطرفية (Terminal rendering)
    # ---------------------------------------------------------
    def render(self):
        lines = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                pos = (r, c)
                if pos == self.agent_pos and pos == self.goal:
                    row.append(SYMBOLS["agent_on_goal"])
                elif pos == self.agent_pos:
                    row.append(SYMBOLS["agent"])
                elif pos in self.walls:
                    row.append(SYMBOLS["wall"])
                elif pos in self.boxes:
                    row.append(SYMBOLS["box"])
                elif pos == self.goal:
                    row.append(SYMBOLS["goal"])
                else:
                    row.append(SYMBOLS["empty"])
            lines.append(" ".join(row))
        return "\n".join(lines)

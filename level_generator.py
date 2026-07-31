"""
level_generator.py
-------------------
يولّد مراحل عشوائية جديدة (جدران، صناديق، هدف، نقطة بداية)
ويتأكد أن كل مرحلة **قابلة للحل** قبل تسليمها للوكيل، عبر بحث
BFS من نقطة البداية إلى الهدف.

هذا هو المسؤول عن فكرة "لما يحل مرحلة، تجيه مرحلة جديدة عشوائياً".
"""

import random
from collections import deque


def _reachable(start, goal, blocked, width, height):
    """
    فحص وصول بسيط عبر BFS: هل يوجد مسار من start إلى goal
    بدون المرور بأي خلية في `blocked`؟

    ملاحظة مهمة: نمرّر هنا (جدران + صناديق) معاً كـ `blocked`،
    أي نتأكد من وجود مسار **لا يحتاج إطلاقاً** لدفع أي صندوق.
    لو اعتبرنا الصناديق قابلة للعبور بحرية في الفحص، قد نولّد
    مرحلة يكون فيها الصندوق عالقاً في ممر ضيق بحيث يستحيل دفعه
    فعلياً أثناء اللعب -> مرحلة مستحيلة الحل. بجعل الصندوق حاجزاً
    في الفحص، نضمن أن هناك دائماً طريقاً حول الصندوق حتى لو لم
    يستطع الوكيل دفعه أبداً.
    """
    if start == goal:
        return True

    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            npos = (r + dr, c + dc)
            nr, nc = npos
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if npos in blocked or npos in visited:
                continue
            if npos == goal:
                return True
            visited.add(npos)
            queue.append(npos)
    return False


def generate_level(width, height, wall_density=0.15, num_boxes=1,
                    rng=None, max_tries=300):
    """
    يولّد مرحلة عشوائية جديدة ويرجع:
        (walls, boxes, goal, agent_start)

    - إطار خارجي دائماً جدار (حدود العالم).
    - جدران داخلية عشوائية بنسبة wall_density تقريباً.
    - يتأكد عبر BFS أن الهدف قابل للوصول من نقطة البداية،
      وإلا يعيد المحاولة بتوليد آخر (حتى max_tries مرة).
    - في حال فشل كل المحاولات (نادر جداً)، يرجع مرحلة بسيطة
      مضمونة الحل (بدون جدران داخلية).
    """
    rng = rng or random.Random()

    interior = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
    ]

    for _ in range(max_tries):
        walls = set()
        for c in range(width):
            walls.add((0, c))
            walls.add((height - 1, c))
        for r in range(height):
            walls.add((r, 0))
            walls.add((r, width - 1))

        n_interior_walls = int(len(interior) * wall_density)
        candidate_walls = rng.sample(interior, min(n_interior_walls, len(interior)))
        walls.update(candidate_walls)

        free_cells = [p for p in interior if p not in walls]
        if len(free_cells) < num_boxes + 2:
            continue

        agent_start = rng.choice(free_cells)
        remaining = [p for p in free_cells if p != agent_start]

        goal = rng.choice(remaining)
        remaining = [p for p in remaining if p != goal]

        if len(remaining) < num_boxes:
            continue
        boxes = rng.sample(remaining, num_boxes)

        blocked = walls | set(boxes)
        if _reachable(agent_start, goal, blocked, width, height):
            return walls, boxes, goal, agent_start

    # مرحلة احتياطية مضمونة الحل (بدون عوائق داخلية)
    walls = set()
    for c in range(width):
        walls.add((0, c))
        walls.add((height - 1, c))
    for r in range(height):
        walls.add((r, 0))
        walls.add((r, width - 1))
    agent_start = (1, 1)
    goal = (height - 2, width - 2)
    boxes = [(height // 2, width // 2)]
    return walls, boxes, goal, agent_start

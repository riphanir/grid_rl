"""
progress.py
-----------
يحفظ ويحمّل إحصائيات تراكمية عن تقدّم التدريب عبر تشغيلات
متعددة (مثلاً كل مرة يشتغل فيها GitHub Action). هذا منفصل عن
جدول Q نفسه (الذي يحفظه agent.py) لأنه معلومات وصفية فقط.
"""

import json
import os


DEFAULT_STATS = {
    "total_runs": 0,
    "total_episodes": 0,
    "total_levels_solved": 0,
    "last_level_number": 1,
}


def load_stats(path):
    """يحمّل الإحصائيات المحفوظة، أو يرجع قيماً ابتدائية إن لم يوجد ملف."""
    if not os.path.exists(path):
        return dict(DEFAULT_STATS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(DEFAULT_STATS)
        merged.update(data)
        return merged
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_STATS)


def save_stats(path, stats):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

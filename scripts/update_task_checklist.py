"""修复任务清单HTML — 先重置所有标记，再正确标记（向前搜索task-check）"""
from pathlib import Path

html_path = Path(__file__).parent.parent / "fall-risk-tech-tasks.html"
html = html_path.read_text(encoding="utf-8")

# 1. 重置所有 task-check（去掉 checked/partial 类名）
import re
html = re.sub(
    r'<div class="task-check(?:\s+(?:checked|partial))?"></div>',
    '<div class="task-check"></div>',
    html,
)

# 2. 定义任务状态
done = ["T1.1", "T1.4"]
partial = ["T1.2", "T1.5", "T2.2", "T2.3", "T3.1", "T3.2", "T3.3",
           "T7.1", "T8.1", "T8.2", "T8.3", "T11.1", "T12.1"]

# 3. 对每个任务，找到 task-id 位置，向前找最近的 task-check
old_check = '<div class="task-check"></div>'
for tid in done + partial:
    cls = "checked" if tid in done else "partial"
    new_check = f'<div class="task-check {cls}"></div>'

    marker = f'task-id">{tid}</span>'
    idx = html.find(marker)
    if idx < 0:
        print(f"  [WARN] 未找到 {tid}")
        continue

    # 向前搜索最近的 task-check
    check_idx = html.rfind(old_check, 0, idx)
    if check_idx < 0:
        print(f"  [WARN] {tid} 前未找到 task-check")
        continue

    html = html[:check_idx] + new_check + html[check_idx + len(old_check):]
    print(f"  [OK] {tid} -> {cls}")

# 4. 写回
html_path.write_text(html, encoding="utf-8")
print("\n修复完成！")

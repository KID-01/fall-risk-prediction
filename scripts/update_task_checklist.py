"""更新任务清单标记 — 2026-07-15 第二批"""
from pathlib import Path
import re

html_path = Path(__file__).parent.parent / "docs" / "fall-risk-tech-tasks.html"
html = html_path.read_text(encoding="utf-8")

# 重置
html = re.sub(r'<div class="task-check(?:\s+(?:checked|partial))?"></div>', '<div class="task-check"></div>', html)

done = [
    "T1.1", "T1.4", "T1.5", "T2.1", "T2.2", "T2.3", "T2.5",
    "T5.1", "T5.2", "T5.3", "T5.4", "T5.5",
    "T7.3",   # WebSocket推送
    "T8.1",   # FastAPI路由拆分
    "T8.2",   # 推理服务封装
    "T8.3",   # 数据持久化(数据库层)
    "T9.1",   # 家属端实时看板
    "T11.4",  # Docker部署
]
partial = [
    "T1.2", "T3.1", "T3.2", "T3.3", "T7.1", "T12.1",
]

old_check = '<div class="task-check"></div>'
for tid in done + partial:
    cls = "checked" if tid in done else "partial"
    new_check = f'<div class="task-check {cls}"></div>'
    marker = f'task-id">{tid}</span>'
    idx = html.find(marker)
    if idx < 0: continue
    check_idx = html.rfind(old_check, 0, idx)
    if check_idx < 0: continue
    html = html[:check_idx] + new_check + html[check_idx + len(old_check):]
    print(f"  [OK] {tid} -> {cls}")

html_path.write_text(html, encoding="utf-8")

# ── 更新 Markdown 大纲 ──
md_path = Path(__file__).parent.parent / "docs" / "挑战杯大纲0.1.md"
content = md_path.read_text(encoding="utf-8")
content = re.sub(r'\s*\[✅\]|\s*\[◐\]', '', content)

lines = content.split("\n")
new_lines = []
for line in lines:
    m = re.match(r'^(### T\d+\.\d+ )', line)
    if m:
        tid = line.split()[1]
        if tid in done:
            line = line + " [✅]"
        elif tid in partial:
            line = line + " [◐]"
    new_lines.append(line)
md_path.write_text("\n".join(new_lines), encoding="utf-8")

done_count = sum(1 for l in new_lines if "[✅]" in l)
partial_count = sum(1 for l in new_lines if "[◐]" in l)
print(f"\n标记完成: ✅{done_count}个已完成, ◐{partial_count}个部分完成")

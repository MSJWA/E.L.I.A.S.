# gantt_chart_fixed.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ---------- tasks ----------
tasks = [
    ("Setup & Folder Structure", "2024-11-05", "2024-11-05"),
    ("NLP Core (Regex)", "2024-11-06", "2024-11-07"),
    ("Vision Placeholder", "2024-11-08", "2024-11-09"),
    ("Plugin Setup", "2024-11-09", "2024-11-10"),
    ("Documentation Setup", "2024-11-09", "2024-11-10"),
    ("App Integration", "2024-11-11", "2024-11-12"),
    ("NLP Hybrid Model", "2024-11-11", "2024-11-14"),
    ("Plugin Development", "2024-11-13", "2024-11-15"),
    ("Vision Detection", "2024-11-13", "2024-11-17"),
    ("Integrate NLP ↔ Plugins", "2024-11-18", "2024-11-19"),
    ("Extend Intents", "2024-11-18", "2024-11-22"),
    ("Streamlit UI", "2024-11-20", "2024-11-23"),
    ("Enhance Plugins", "2024-11-20", "2024-11-24"),
    ("Vision Auth Integration", "2024-11-21", "2024-11-24"),
    ("Full System Testing", "2024-11-25", "2024-11-28"),
    ("Error Handling", "2024-11-25", "2024-11-27"),
    ("Explainability Layer", "2024-11-26", "2024-11-29"),
    ("Final Documentation", "2024-11-28", "2024-12-02"),
    ("Rehearsal + Demo Prep", "2024-12-01", "2024-12-03"),
]

# ---------- convert dates ----------
def to_num(date_str):
    return mdates.date2num(datetime.strptime(date_str, "%Y-%m-%d"))

task_names = [t[0] for t in tasks]
start_nums = [to_num(t[1]) for t in tasks]
end_nums   = [to_num(t[2]) for t in tasks]
durations  = [e - s for s, e in zip(start_nums, end_nums)]

# ---------- adjust durations to fit text ----------
fig, ax = plt.subplots(figsize=(14, max(6, len(tasks)*0.35)))

y_pos = list(range(len(tasks)))[::-1]  # reverse so first task is on top

# approximate text width in days (rough estimation)
avg_char_width = 0.25  # in days; adjust if needed

for i, (name, s, dur) in enumerate(zip(task_names[::-1], start_nums[::-1], durations[::-1])):
    text_width = len(name) * avg_char_width
    # extend duration if text is wider than bar
    if text_width > dur:
        dur = text_width
    
    ax.barh(i, dur, left=s, height=0.6, align='center', edgecolor='black', color="#7EC8E3")
    ax.text(s + dur/2, i, name, va='center', ha='center', fontsize=8, color='black')

# ensure x-limits show all bars with a bit of margin
xmin = min(start_nums) - 1
xmax = max(end_nums + durations) + 2  # leave margin
ax.set_xlim(xmin, xmax)

# aesthetics
ax.set_yticks([])  # remove y-axis ticks
ax.set_xlabel("Date")
ax.set_title("ELIAS — Gantt Chart (5 Nov – 3 Dec)")

# format x-axis dates
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=45)

plt.tight_layout()

outfile = "gantt_chart.png"
plt.savefig(outfile, dpi=200)
print("Saved:", outfile)
plt.show()

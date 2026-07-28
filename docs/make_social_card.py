"""
Agent Memory Engine — social preview card (1200x630).

The image used as the GitHub repo's social preview (Settings -> Social
preview) and as an OG image if a docs/demo site is added later.

Regenerate:
    python docs/make_social_card.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'

# GitHub dark palette (pops in social feeds)
BG = '#0d1117'
FG = '#f0f6fc'
MUTED = '#8b949e'
BLUE = '#58a6ff'
BLUE_DEEP = '#388bfd'
ORANGE = '#e69138'
GREEN = '#3fb950'
DIVIDER = '#30363d'

fig, ax = plt.subplots(figsize=(12, 6.3), dpi=100)   # 1200 x 630
ax.set_xlim(0, 120)
ax.set_ylim(0, 63)
ax.axis('off')
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)


def chip(x, y, w, h, fc, text, tc=FG, fs=11, bold=True):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.8',
                       fc=fc, ec='none')
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, weight='bold' if bold else 'normal')


# ===== monogram (top-left) =====
chip(8, 53.5, 9, 6, BLUE, 'ame', fs=15)

# ===== title =====
ax.text(8, 47, 'Agent Memory Engine', fontsize=46, color=FG, weight='bold')
ax.text(8, 41.5, 'Long-term memory for coding agents',
        fontsize=17, color=MUTED)

# ===== features (bullet rows) =====
features = [
    ('Two-stage retrieval + gating', BLUE),
    ('Hybrid recall: vector + BM25 (RRF)', BLUE_DEEP),
    ('Cross-encoder precision rerank', ORANGE),
    ('Single SQLite file, zero ops', GREEN),
]
fy = 35
for text, color in features:
    ax.plot(9, fy, 'o', color=color, markersize=9)
    ax.text(12, fy, text, fontsize=14.5, color=FG, va='center')
    fy -= 4.3

# ===== url (bottom-left) =====
ax.text(8, 5, 'github.com/ljftwq-dev/agent-memory-engine',
        fontsize=12.5, color=MUTED)

# ===== vertical divider =====
ax.plot([69, 69], [9, 55], color=DIVIDER, lw=1.2)

# ===== right side: benchmark mini bar chart =====
ax.text(94, 56, 'Retrieval quality  ·  nDCG@5',
        fontsize=13.5, color=MUTED, ha='center', weight='bold')

labels = ['pure\nvector', '+BM25\nhybrid', '+rerank']
vals = [0.842, 0.868, 0.927]
colors = [BLUE, BLUE_DEEP, ORANGE]
xs = [80, 94, 108]
bw = 8.5
base_y = 12
# baseline
ax.plot([74, 116], [base_y, base_y], color=DIVIDER, lw=1.2)

for x, v, c, lab in zip(xs, vals, colors, labels):
    h = (v - 0.80) * 175          # zoom in on the 0.80-0.95 band
    p = FancyBboxPatch((x - bw / 2, base_y), bw, h,
                       boxstyle='round,pad=0.02,rounding_size=0.5',
                       fc=c, ec='none')
    ax.add_patch(p)
    ax.text(x, base_y + h + 1.8, f'{v:.3f}',
            ha='center', va='center', fontsize=13, color=c, weight='bold')
    ax.text(x, base_y - 2.4, lab, ha='center', va='center',
            fontsize=10.5, color=MUTED, linespacing=1.1)

ax.text(94, 6.5, 'each stage earns its keep',
        fontsize=11, color=MUTED, ha='center', style='italic')

_HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(_HERE, 'images', 'social-card.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=100, facecolor=BG)
print('saved:', out)

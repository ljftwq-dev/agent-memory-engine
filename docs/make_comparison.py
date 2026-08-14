"""
Agent Memory Engine — "how it differs" comparison card (us vs Mem0 vs Zep).

A compact feature matrix that makes the architectural tradeoffs legible at a
glance. Honest by design: rows where a competitor is strictly better (e.g.
Zep's first-class temporal queries) are marked as such.

Regenerate:
    python docs/make_comparison.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'

# Light palette (reads well on GitHub's white background + in README).
WHITE = '#ffffff'
NAVY = '#1f3a5f'
COL_HIGHLIGHT = '#eaf2fb'   # our column background
COL_DIM = '#f6f8fa'         # dimension-name column background
GREEN = '#1a7f37'           # a clear advantage
ORANGE = '#cf6a1a'          # a drawback
GRAY = '#57606a'            # neutral text
DIVIDER = '#d0d7de'
HEAD_BG = '#1f3a5f'

# Each cell: (text, mark) where mark in {None, 'good', 'bad'}
# None = neutral, 'good' = green check, 'bad' = orange cross.
COLUMNS = ['Agent Memory Engine', 'Mem0', 'Zep / Graphiti']
ROWS = [
    ('External services',
     ('One SQLite file\n(zero ops)', 'good'),
     ('SDK + a vector\nstore', None),
     ('Graph DB required\n(Neo4j/FalkorDB)', 'bad')),
    ('Write-path cost',
     ('None\n(no LLM to store)', 'good'),
     ('LLM call\nper write', 'bad'),
     ('Entity/edge\nconstruction', 'bad')),
    ('Memory model',
     ('Ebbinghaus decay\n+ hybrid rerank', None),
     ('LLM reconcile\nADD/UPD/DEL', None),
     ('Temporal\nknowledge graph', None)),
    ('Temporal queries\n("when did X change?")',
     ('—', None),
     ('—', None),
     ('First-class\n(valid_at)', 'good')),
    ('Self-host footprint',
     ('Lowest\n(1 process)', 'good'),
     ('Low\n(SDK)', None),
     ('High\n(3 systems)', 'bad')),
    ('License',
     ('MIT', None),
     ('Apache 2.0', None),
     ('Apache 2.0\n(core only)', None)),
]

MARK_GLYPH = {'good': '\u2713', 'bad': '\u2717'}      # check / cross
MARK_COLOR = {'good': GREEN, 'bad': ORANGE}

fig, ax = plt.subplots(figsize=(13, 7.2), dpi=200)
ax.set_xlim(0, 130)
ax.set_ylim(0, 72)
ax.axis('off')
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)


def rbox(x, y, w, h, fc, ec='none', lw=0, radius=0.6):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0.02,rounding_size={radius}',
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)


# ---- Title ----
ax.text(2, 69, 'How it differs', fontsize=23, color=NAVY, weight='bold')
ax.text(2, 65.2,
        'Agent Memory Engine vs Mem0 vs Zep  —  local-first memory for coding agents',
        fontsize=11, color=GRAY)

# ---- Layout geometry ----
LEFT = 2
NAME_W = 26                      # dimension-name column width
COL_W = 33                       # data column width
GAP = 0.6
TOP = 59
ROW_H = 7.2
HEAD_H = 5.0

name_x = LEFT
col_xs = [LEFT + NAME_W + GAP + i * (COL_W + GAP) for i in range(3)]
# highlight our column with a soft band behind it
rbox(col_xs[0] - 0.8, TOP - len(ROWS) * ROW_H - HEAD_H - 1.2,
     COL_W + 1.6, len(ROWS) * ROW_H + HEAD_H + 2.0, COL_HIGHLIGHT, radius=1.0)

# ---- Header row ----
rbox(name_x, TOP - HEAD_H, NAME_W, HEAD_H, HEAD_BG, radius=0.8)
ax.text(name_x + NAME_W / 2, TOP - HEAD_H / 2, '',
        ha='center', va='center', color='white', weight='bold')
for x, title in zip(col_xs, COLUMNS):
    weight = 'bold' if title == COLUMNS[0] else 'bold'
    color = '#ffffff'
    rbox(x, TOP - HEAD_H, COL_W, HEAD_H, HEAD_BG, radius=0.8)
    ax.text(x + COL_W / 2, TOP - HEAD_H / 2, title,
            ha='center', va='center', color=color, fontsize=11, weight=weight,
            linespacing=1.1)

# ---- Body rows ----
for ri, row in enumerate(ROWS):
    name = row[0]
    cells = row[1:]
    y_top = TOP - HEAD_H - ri * ROW_H
    y_bot = y_top - ROW_H
    yc = (y_top + y_bot) / 2

    # dimension name cell
    rbox(name_x, y_bot, NAME_W, ROW_H, COL_DIM, ec=DIVIDER, lw=0.8, radius=0.4)
    ax.text(name_x + 1.5, yc, name, ha='left', va='center',
            fontsize=10.2, color=NAVY, weight='bold', linespacing=1.15)

    for ci, (x, (text, mark)) in enumerate(zip(col_xs, cells)):
        # our column already has a band; others get a thin divider feel
        if ci != 0:
            rbox(x, y_bot, COL_W, ROW_H, WHITE, ec=DIVIDER, lw=0.6, radius=0.4)
        # text
        ax.text(x + COL_W / 2, yc + 0.3, text, ha='center', va='center',
                fontsize=9.6, color=GRAY, linespacing=1.2)
        # mark glyph under the text
        if mark is not None:
            ax.text(x + COL_W / 2, y_bot + 1.0, MARK_GLYPH[mark],
                    ha='center', va='center', fontsize=12,
                    color=MARK_COLOR[mark], weight='bold')

# ---- Footer ----
ax.text(LEFT, 2.5,
        'Three architectures, one category.  We optimize for local, zero-ops, '
        'no write-path LLM cost —  pick by your constraint, not by star count.',
        fontsize=9.5, color=GRAY, style='italic')
ax.text(LEFT + 126, 2.5, 'github.com/ljftwq-dev/agent-memory-engine',
        fontsize=8.5, color=GRAY, ha='right')

_here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(_here, 'images', 'comparison.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, facecolor=WHITE, bbox_inches='tight', pad_inches=0.15)
print('saved:', out)

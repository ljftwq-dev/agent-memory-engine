"""
Agent Memory Engine — architecture diagram (matplotlib version).

Regenerates docs/images/architecture.png from source. Edit the layout / labels
here and re-run to update the diagram that's checked into the repo.

Usage:
    python docs/make_architecture.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(14, 10), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

C_AGENT_F, C_AGENT_E = '#E8EAED', '#9AA0A6'
C_INT_F,   C_INT_E   = '#D6E4F0', '#4A90D9'
C_CORE_F,  C_CORE_E  = '#FFF8EC', '#E69138'
C_PIPE_F              = '#3B7DD8'
C_PIPE2_F             = '#5BA3E0'
C_SCORE_F             = '#E69138'
C_WRITE_F, C_WRITE_E = '#FCE5CD', '#E69138'
C_STOR_F,  C_STOR_E  = '#D9EAD3', '#6AA84F'
C_ARROW                = '#5F6368'


def box(x, y, w, h, fc, ec, text, fs=10, tc='black', lw=1.6, weight='normal'):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.6',
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, weight=weight, linespacing=1.35)


def band(x, y, w, h, fc, ec, lw=1.8):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.8',
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)


def arrow(x1, y1, x2, y2, color=C_ARROW, lw=2.0):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=16,
                        color=color, lw=lw)
    ax.add_patch(a)


def label(x, y, text, fs=9, color='#5F6368', weight='bold', ha='left', style='normal'):
    ax.text(x, y, text, fontsize=fs, color=color, weight=weight,
            ha=ha, va='center', style=style)


# ===== Title =====
ax.text(50, 97.5, 'Agent Memory Engine — Architecture',
        ha='center', fontsize=20, weight='bold')
ax.text(50, 94.3,
        'Long-term memory for coding agents  ·  two-stage retrieval + gating  ·  single SQLite file',
        ha='center', fontsize=10.5, color='#5F6368')

# ===== Agent Host =====
label(6, 91.5, 'AGENT HOST')
box(15, 85.5, 20, 5.5, C_AGENT_F, C_AGENT_E, 'opencode', 11)
box(40, 85.5, 20, 5.5, C_AGENT_F, C_AGENT_E, 'Claude Code', 11)
box(65, 85.5, 20, 5.5, C_AGENT_F, C_AGENT_E, 'any MCP / HTTP agent', 10)
arrow(50, 85.5, 50, 81.5)
ax.text(51, 83.5, 'HTTP / MCP', fontsize=8.5, color=C_ARROW)

# ===== Interface =====
label(6, 80.3, 'INTERFACE')
box(15, 74, 70, 5.5, C_INT_F, C_INT_E,
    'HTTP Server  :8765        ·        recall  /  remember  /  recent  /  search  /  forget',
    11)
arrow(50, 74, 50, 69.5)
ax.text(51, 71.7, 'requests', fontsize=8.5, color=C_ARROW)

# ===== Engine Core (big band) =====
band(4, 18, 92, 50, C_CORE_F, C_CORE_E, lw=2)
label(7, 66.5, 'ENGINE CORE', color=C_CORE_E)
ax.text(93, 66.5,
        'owns the Episodic layer of the 4-layer memory model',
        fontsize=8.5, color='#999', ha='right', style='italic')

# ----- RETRIEVE pipeline -----
label(7, 62.5, 'RETRIEVE   —   GET /recall', color='#B45309', fs=10.5)
pipe_labels = [
    ('Stage A\nWide recall', 'vec KNN · 15\n+ BM25 · 15'),
    ('Gate\ndrop noise', 'distance\n> threshold'),
    ('Fuse\nRRF', 'vector + BM25\nrank fusion'),
    ('Rerank\ncross-encoder', 'optional\nbge-reranker-v2-m3'),
    ('Score\nfinal rank', 'α·strength\n+ (1-α)·relevance'),
]
pipe_fill = [C_PIPE_F, C_PIPE2_F, C_PIPE2_F, C_PIPE2_F, C_SCORE_F]
xs = [6, 24, 42, 60, 78]
w = 16
for x, (t, sub), fc in zip(xs, pipe_labels, pipe_fill):
    box(x, 51, w, 10, fc, '#2C6FB0', '', lw=1.4)
    ax.text(x + w / 2, 57.5, t, ha='center', va='center',
            fontsize=9.5, color='white', weight='bold', linespacing=1.2)
    ax.text(x + w / 2, 53, sub, ha='center', va='center',
            fontsize=7.8, color='white', linespacing=1.2)
for i in range(4):
    arrow(xs[i] + w, 56, xs[i + 1], 56, lw=1.6)
arrow(2.5, 56, 6, 56)
ax.text(2, 56, 'query', fontsize=9, ha='right', va='center', style='italic')
arrow(xs[4] + w, 56, xs[4] + w + 3.2, 56)
ax.text(xs[4] + w + 3.5, 56, 'top-k', fontsize=9, va='center', style='italic')
ax.text(50, 48.5,
        'strength = exp(−Δt / τ)   ·   every recall does τ *= 1.5  (Ebbinghaus decay, no RL needed)',
        ha='center', fontsize=8.8, color='#B45309', style='italic')

# ----- WRITE paths -----
label(7, 45, 'WRITE   —   POST /remember  ·  POST /forget', color='#B45309', fs=10.5)

wb_y = 38
arrow(21, wb_y + 2.5, 22.5, wb_y + 2.5, lw=1.5)
arrow(38.5, wb_y + 2.5, 40, wb_y + 2.5, lw=1.5)
arrow(56, wb_y + 2.5, 57.5, wb_y + 2.5, lw=1.5)
box(9, wb_y, 12, 5, C_WRITE_F, C_WRITE_E, 'POST\n/remember', 9, weight='bold')
box(22.5, wb_y, 16, 5, C_WRITE_F, C_WRITE_E, 'LLM summary\n(optional)', 9)
box(40, wb_y, 16, 5, C_WRITE_F, C_WRITE_E, 'dedup-merge\n(dist ≤ thr)', 9)
box(57.5, wb_y, 20, 5, C_STOR_F, C_STOR_E, 'store → episodic', 9, weight='bold')

fb_y = 28
arrow(21, fb_y + 2.5, 22.5, fb_y + 2.5, lw=1.5)
arrow(46.5, fb_y + 2.5, 48, fb_y + 2.5, lw=1.5)
arrow(64, fb_y + 2.5, 65.5, fb_y + 2.5, lw=1.5)
box(9, fb_y, 12, 5, C_WRITE_F, C_WRITE_E, 'POST\n/forget', 9, weight='bold')
box(22.5, fb_y, 24, 5, C_WRITE_F, C_WRITE_E, 'Ebbinghaus decay\nτ *= 1.5 per recall', 9)
box(48, fb_y, 16, 5, C_WRITE_F, C_WRITE_E, 'purge weak\nstrength < min', 9)
box(65.5, fb_y, 16, 5, C_STOR_F, C_STOR_E, 'rebalance', 9, weight='bold')

# ===== arrow to storage =====
arrow(50, 18, 50, 15)
ax.text(51, 16.5, 'sqlite-vec KNN  +  FTS5 BM25', fontsize=8.5, color=C_ARROW)

# ===== Storage =====
label(6, 14.2, 'STORAGE  —  single SQLite file (memory.db)', color=C_STOR_E)
box(12, 4.5, 36, 8.5, C_STOR_F, C_STOR_E,
    'episodic   (table)\ntopic · summary · raw\nstrength · tau\nlast_recall_ts · created_ts', 9)
box(52, 4.5, 32, 8.5, C_STOR_F, C_STOR_E,
    'episodic_vec   (vec0 virtual)\nFLOAT[dim] embedding\njoined by rowid', 9)

# ===== caption =====
ax.text(50, 1.5,
        '4-layer memory model:  working · episodic · semantic · procedural'
        '   —   this engine owns only the Episodic layer; the rest live in your agent host.',
        ha='center', fontsize=8.8, color='#777', style='italic')

_here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(_here, 'images', 'architecture.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, facecolor='white')
print('saved:', out)

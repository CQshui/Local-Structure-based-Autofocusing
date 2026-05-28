"""
File: simple_error_analysis_final.py
Purpose: Publication-quality error analysis with boxplot and bar chart with error bars
         Optimized for academic papers with 1:1 square format
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm

# ==============================
# 1. Font Configuration for Times New Roman
# ==============================
def setup_times_new_roman():
    """Setup Times New Roman font with fallback options"""

    available_fonts = {f.name.lower(): f.name for f in fm.fontManager.ttflist}

    times_variants = [
        'Times New Roman',
        'Times',
        'Liberation Serif',
        'DejaVu Serif',
        'FreeSerif'
    ]

    font_found = None
    for variant in times_variants:
        if variant.lower() in available_fonts.values() or any(variant.lower() in f.lower() for f in available_fonts.keys()):
            font_found = variant
            break

    if font_found:
        print(f"✓ Using font: {font_found}")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [font_found]
    else:
        print("⚠ Times New Roman not found. Using DejaVu Serif fallback")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif']

    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    return font_found or 'DejaVu Serif'

font_name = setup_times_new_roman()

# ==============================
# 2. Enhanced Publication Settings
# ==============================
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10

plt.rcParams.update({
    # Font settings
    'font.size': FONT_SIZE_LABEL,
    'axes.titlesize': FONT_SIZE_TITLE,
    'axes.labelsize': FONT_SIZE_LABEL,
    'xtick.labelsize': FONT_SIZE_TICK,
    'ytick.labelsize': FONT_SIZE_TICK,
    'legend.fontsize': FONT_SIZE_LEGEND,
    'figure.titlesize': FONT_SIZE_TITLE,

    # Line widths
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.0,
    'grid.linewidth': 0.6,

    # Grid styling
    'grid.alpha': 0.25,
    'grid.linestyle': ':',
    'grid.color': '#CCCCCC',

    # Ticks
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',

    # Figure
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,

    # Legend
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#666666',
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.borderpad': 0.5,
})

# Professional color palette
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

# ==============================
# 3. Configuration
# ==============================
# folder_path = Path(r"F:\dongjiayao\Data\AutoFocusDatabase\article\comparison\0-CSV")
# folder_path = Path(r"F:\dongjiayao\Data\AutoFocusDatabase\article\comparison\000")
# folder_path = Path(r"F:\dongjiayao\Data\COCO\val\article\comparison\0-CSV")
folder_path = Path(r"F:\dongjiayao\Data\COCO\val\article\comparison\000")

if not folder_path.exists():
    folder_path = Path(".")
    print(f"⚠ Using current directory: {folder_path.absolute()}")
else:
    print(f"✓ Input folder: {folder_path}")

FIRST_FILE_NAME = "ours"
OUTPUT_DPI = 1200
MAX_SIZE_CM = 8.5  # Square: 8.5 cm × 8.5 cm
MAX_SIZE_INCHES = MAX_SIZE_CM / 2.54

print(f"📐 Square format: {MAX_SIZE_CM} cm × {MAX_SIZE_CM} cm ({MAX_SIZE_INCHES:.2f}\" × {MAX_SIZE_INCHES:.2f}\")")

# ==============================
# 4. Data Loading
# ==============================
csv_files = list(folder_path.glob("*.csv"))

if not csv_files:
    print(f"❌ Error: No CSV files found in {folder_path}")
    exit()

print(f"✓ Found {len(csv_files)} CSV files")

# Sort files
first_file = None
other_files = []

for csv_file in csv_files:
    if csv_file.stem == FIRST_FILE_NAME:
        first_file = csv_file
    else:
        other_files.append(csv_file)

if first_file:
    csv_files_sorted = [first_file] + sorted(other_files, key=lambda x: x.name)
    print(f"✓ '{FIRST_FILE_NAME}' will be plotted first")
else:
    csv_files_sorted = sorted(csv_files, key=lambda x: x.name)
    print(f"⚠ '{FIRST_FILE_NAME}.csv' not found, using alphabetical order")

method_data = {}
method_stats = {}

for csv_file in csv_files_sorted:
    method_name = csv_file.stem
    print(f"\n📊 Processing: {method_name}")

    try:
        try:
            df = pd.read_csv(csv_file)
        except:
            with open(csv_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()

            if ',' in first_line:
                column_names = [col.strip() for col in first_line.split(',')]
                df = pd.read_csv(csv_file, skiprows=1, header=None)

                if df.shape[1] == 1:
                    df = df[0].str.split(',', expand=True)

                if df.shape[1] == len(column_names):
                    df.columns = column_names

        abs_error_col = None
        for col in df.columns:
            if 'abs_error' in col.lower() or 'error' in col.lower():
                abs_error_col = col
                break

        if abs_error_col is None:
            print(f"  ❌ No error column found")
            continue

        abs_errors = pd.to_numeric(df[abs_error_col], errors='coerce').dropna()

        if len(abs_errors) == 0:
            print(f"  ❌ No valid data")
            continue

        method_data[method_name] = abs_errors

        method_stats[method_name] = {
            'count': len(abs_errors),
            'mean': float(abs_errors.mean()),
            'std': float(abs_errors.std()),
            'min': float(abs_errors.min()),
            'max': float(abs_errors.max()),
            'median': float(abs_errors.median()),
            'q1': float(abs_errors.quantile(0.25)),
            'q3': float(abs_errors.quantile(0.75)),
            'sem': float(abs_errors.sem()),  # Standard Error of Mean
        }

        print(f"  ✓ Loaded {len(abs_errors)} points (μ={abs_errors.mean()*1e6:.2f} μm, σ={abs_errors.std()*1e6:.2f} μm)")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue

if not method_data:
    print("\n❌ No valid data loaded. Exiting.")
    exit()

print(f"\n✓ Successfully loaded {len(method_data)} methods")

# ==============================
# 5. Enhanced Square Boxplot
# ==============================
print("\n📈 Generating enhanced boxplot (1:1 ratio)...")

fig1, ax1 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))

method_names = list(method_data.keys())
# Convert to micrometers for display
data_to_plot = [method_data[name].values * 1e6 for name in method_names]

# Create enhanced boxplot
bp = ax1.boxplot(data_to_plot,
                 labels=method_names,
                 patch_artist=True,
                 showmeans=True,
                 showfliers=False,
                 widths=0.6,
                 medianprops=dict(color='#2C3E50', linewidth=2.0, solid_capstyle='round'),
                 meanprops=dict(marker='D', markerfacecolor='#E74C3C',
                               markeredgecolor='#C0392B', markersize=5,
                               markeredgewidth=1.0, zorder=3),
                 whiskerprops=dict(color='black', linewidth=1.2, linestyle='-'),
                 capprops=dict(color='black', linewidth=1.2),
                 boxprops=dict(linewidth=1.2, edgecolor='#2C3E50'))

# Enhanced color scheme
color_list = list(COLORS.values())
for patch, i in zip(bp['boxes'], range(len(method_names))):
    color = color_list[i % len(color_list)]
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
    patch.set_edgecolor('#2C3E50')

# Enhanced labels
ax1.set_ylabel('Absolute Error (μm)', fontweight='semibold', labelpad=8)
ax1.set_xlabel('Method', fontweight='semibold', labelpad=8)
ax1.set_title('Error Distribution Comparison', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE+1)

# Enhanced grid
ax1.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.8, color='#AAAAAA')
ax1.set_axisbelow(True)

# Enhanced x-axis labels
if len(method_names) > 5:
    ax1.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK-0.5)
    plt.setp(ax1.xaxis.get_majorticklabels(), ha='right', rotation_mode='anchor')
else:
    ax1.tick_params(axis='x', rotation=0)

# Add subtle background
ax1.set_facecolor('#FAFAFA')

# Enhanced spine visibility
for spine in ax1.spines.values():
    spine.set_linewidth(1.2)
    spine.set_edgecolor('#666666')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2C3E50', linewidth=2, label='Median'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#E74C3C',
           markeredgecolor='#C0392B', markersize=5, label='Mean', linestyle='None')
]
ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.95,
          edgecolor='#666666', fancybox=False)

plt.tight_layout(pad=0.5)

boxplot_file = folder_path / "error_boxplot_square.png"
plt.savefig(boxplot_file, dpi=OUTPUT_DPI, bbox_inches='tight', pad_inches=0.1)
print(f"✓ Saved: {boxplot_file.name}")

plt.savefig(folder_path / "error_boxplot_square.pdf", bbox_inches='tight', pad_inches=0.1)
print(f"✓ Saved: error_boxplot_square.pdf")

plt.show()
plt.close(fig1)

# ==============================
# 6. Bar Chart with Error Bars (Mean ± SD)
# ==============================
print("\n📊 Generating bar chart with error bars (1:1 ratio)...")

fig2, ax2 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))

# Prepare data - convert to micrometers
x_pos = np.arange(len(method_names))
means = [method_stats[name]['mean'] * 1e6 for name in method_names]
stds = [method_stats[name]['std'] * 1e6 for name in method_names]

# Create bar chart with error bars
bars = ax2.bar(x_pos, means, yerr=stds,
               color=[color_list[i % len(color_list)] for i in range(len(method_names))],
               alpha=0.8,
               edgecolor='#2C3E50',
               linewidth=1.2,
               capsize=5,
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capthick': 1.5})

# Add value labels on top of bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + std,
             f'{mean:.1f}',
             ha='center', va='bottom', fontsize=FONT_SIZE_TICK-1,
             fontweight='semibold', color='#2C3E50')

# Labels and title
ax2.set_ylabel('Mean Absolute Error ± SD (μm)', fontweight='semibold', labelpad=8)
ax2.set_xlabel('Method', fontweight='semibold', labelpad=8)
ax2.set_title('Mean Error with Standard Deviation', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE+1)

# Set x-axis
ax2.set_xticks(x_pos)
ax2.set_xticklabels(method_names)

# Enhanced x-axis labels
if len(method_names) > 5:
    ax2.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK-0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), ha='right', rotation_mode='anchor')
else:
    ax2.tick_params(axis='x', rotation=0)

# Enhanced grid
ax2.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.8, color='#AAAAAA')
ax2.set_axisbelow(True)

# Background
ax2.set_facecolor('#FAFAFA')

# Enhanced spines
for spine in ax2.spines.values():
    spine.set_linewidth(1.2)
    spine.set_edgecolor('#666666')

# Set y-axis to start from 0
ax2.set_ylim(bottom=0)

plt.tight_layout(pad=0.5)

barchart_file = folder_path / "error_barchart_square.png"
plt.savefig(barchart_file, dpi=OUTPUT_DPI, bbox_inches='tight', pad_inches=0.1)
print(f"✓ Saved: {barchart_file.name}")

plt.savefig(folder_path / "error_barchart_square.pdf", bbox_inches='tight', pad_inches=0.1)
print(f"✓ Saved: error_barchart_square.pdf")

plt.show()
plt.close(fig2)

# ==============================
# 7. Bar Chart with Error Bars (Mean ± SEM)
# ==============================
print("\n📊 Generating bar chart with SEM error bars (1:1 ratio)...")

fig3, ax3 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))

# Prepare data - convert to micrometers
sems = [method_stats[name]['sem'] * 1e6 for name in method_names]

# Create bar chart with SEM error bars
bars = ax3.bar(x_pos, means, yerr=sems,
               color=[color_list[i % len(color_list)] for i in range(len(method_names))],
               alpha=0.8,
               edgecolor='#2C3E50',
               linewidth=1.2,
               capsize=5,
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capthick': 1.5})

# Add value labels on top of bars
for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + sem,
             f'{mean:.1f}',
             ha='center', va='bottom', fontsize=FONT_SIZE_TICK-1,
             fontweight='semibold', color='#2C3E50')

# Labels and title
ax3.set_ylabel('Mean Absolute Error ± SEM (μm)', fontweight='semibold', labelpad=8)
ax3.set_xlabel('Method', fontweight='semibold', labelpad=8)
ax3.set_title('Mean Error with Standard Error', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE+1)

# Set x-axis
ax3.set_xticks(x_pos)
ax3.set_xticklabels(method_names)

# Enhanced x-axis labels
if len(method_names) > 5:
    ax3.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK-0.5)
    plt.setp(ax3.xaxis.get_majorticklabels(), ha='right', rotation_mode='anchor')
else:
    ax3.tick_params(axis='x', rotation=0)

# Enhanced grid
ax3.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.8, color='#AAAAAA')
ax3.set_axisbelow(True)

# Background
ax3.set_facecolor('#FAFAFA')

# Enhanced spines
for spine in ax3.spines.values():
    spine.set_linewidth(1.2)
    spine.set_edgecolor('#666666')

# Set y-axis to start from 0
ax3.set_ylim(bottom=0)

# Add note about SEM
ax3.text(0.02, 0.98, 'SEM = SD / √n', transform=ax3.transAxes,
         fontsize=FONT_SIZE_LEGEND-1, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#666666'))

plt.tight_layout(pad=0.5)

barchart_sem_file = folder_path / "error_barchart_sem_square.png"
plt.savefig(barchart_sem_file, dpi=OUTPUT_DPI, bbox_inches='tight', pad_inches=0.1)
print(f"✓ Saved: {barchart_sem_file.name}")

plt.savefig(folder_path / "error_barchart_sem_square.pdf", bbox_inches='tight', pad_inches=0.1)
print(f"✓ Saved: error_barchart_sem_square.pdf")

plt.show()
plt.close(fig3)

# ==============================
# 8. Statistical Report
# ==============================
print("\n📋 Generating statistical report...")

report_data = []
for method_name, stats in method_stats.items():
    report_data.append({
        'Method': method_name,
        'Count': stats['count'],
        'Mean (μm)': f"{stats['mean']*1e6:.2f}",
        'Std (μm)': f"{stats['std']*1e6:.2f}",
        'SEM (μm)': f"{stats['sem']*1e6:.2f}",
        'Min (μm)': f"{stats['min']*1e6:.2f}",
        'Q1 (μm)': f"{stats['q1']*1e6:.2f}",
        'Median (μm)': f"{stats['median']*1e6:.2f}",
        'Q3 (μm)': f"{stats['q3']*1e6:.2f}",
        'Max (μm)': f"{stats['max']*1e6:.2f}",
        'IQR (μm)': f"{(stats['q3'] - stats['q1'])*1e6:.2f}",
        'CV': f"{stats['std']/stats['mean']:.4f}"
    })

report_df = pd.DataFrame(report_data)
report_file = folder_path / "error_statistics_report.csv"
report_df.to_csv(report_file, index=False)
print(f"✓ Saved: {report_file.name}")

# Enhanced summary table
print("\n" + "="*110)
print("STATISTICAL SUMMARY (Sorted by Mean Error)")
print("="*110)
print(f"{'Rank':<5} {'Method':<20} {'Mean (μm)':<14} {'Std (μm)':<14} {'SEM (μm)':<14} {'Median (μm)':<14} {'CV':<10}")
print("-"*110)

sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]['mean'])
for rank, (method_name, stats) in enumerate(sorted_methods, 1):
    cv = stats['std'] / stats['mean']
    print(f"{rank:<5} {method_name:<20} {stats['mean']*1e6:<14.2f} {stats['std']*1e6:<14.2f} "
          f"{stats['sem']*1e6:<14.2f} {stats['median']*1e6:<14.2f} {cv:<10.4f}")

print("="*110)
print("\n" + "="*110)
print("✅ ANALYSIS COMPLETE - SQUARE FORMAT WITH ERROR BARS")
print("="*110)
print("Generated files:")
print(f"  1. {boxplot_file.name} (Boxplot with median & mean)")
print(f"  2. error_boxplot_square.pdf (vector)")
print(f"  3. {barchart_file.name} (Bar chart with ± SD)")
print(f"  4. error_barchart_square.pdf (vector)")
print(f"  5. {barchart_sem_file.name} (Bar chart with ± SEM)")
print(f"  6. error_barchart_sem_square.pdf (vector)")
print(f"  7. {report_file.name}")
print(f"\nFigure specifications:")
print(f"  • All figures: {MAX_SIZE_CM} cm × {MAX_SIZE_CM} cm (1:1 square)")
print(f"  • Resolution: {OUTPUT_DPI} DPI (ultra-high quality)")
print(f"  • Font: {font_name}")
print(f"  • Format: PNG + PDF (use PDF for LaTeX)")
print(f"  • Units: Micrometers (μm) - all values ×10⁶")
print(f"  • Error bars: BLACK color for all charts")
print(f"\nError bar types:")
print(f"  • SD (Standard Deviation): Shows data variability")
print(f"  • SEM (Standard Error of Mean): Shows estimate precision")
print("="*110)
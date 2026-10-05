from pathlib import Path
import shutil
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import argparse
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Plot the three-seed KITTI source-density results.")
parser.add_argument("--input", type=Path, default=ROOT/"results/retest_20260927/kitti_density_three_seed_summary.csv")
parser.add_argument("--output-dir", type=Path, default=ROOT/"output/reproduced_figures")
args = parser.parse_args()
SOURCE = args.input
OUTPUT = args.output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(SOURCE)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':0.7})
fig, axes = plt.subplots(1,2,figsize=(7.15,2.65),layout='constrained')
colors = {'baseline':'#6B7280','mspki':'#0072B2'}
for ax, metric, ylabel, title in zip(axes,('IR','TR'),('Inlier ratio (%)','Registration success (%)'),('(a) Fine correspondence quality','(b) Pose acceptance')):
    for model, label, marker in (('baseline','Baseline','o'),('mspki','MSPKI','s')):
        rows=data[data.model==model].sort_values('retention_pct',ascending=False)
        x=rows.retention_pct.to_numpy()
        mean=rows[metric+'_pct_mean'].to_numpy()
        sd=rows[metric+'_pct_sd'].to_numpy()
        ax.plot(x,mean,color=colors[model],marker=marker,markersize=4.3,linewidth=1.6,label=label,zorder=3)
        ax.fill_between(x,mean-sd,mean+sd,color=colors[model],alpha=0.13,linewidth=0,zorder=1)
    ax.set_title(title,loc='left',pad=8)
    ax.set_xlabel('Source points retained (%)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(104,21)
    ax.set_xticks([100,75,50,25])
    ax.grid(axis='y',color='#D9DDE1',linewidth=0.6,zorder=0)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylim(42,81)
axes[0].set_yticks([45,55,65,75])
axes[1].set_ylim(94.7,100.4)
axes[1].set_yticks([95,96,97,98,99,100])
axes[0].legend(loc='lower left',frameon=False)
axes[1].legend(loc='lower left',frameon=False)
png=OUTPUT/'kitti_density.png'
fig.savefig(png,dpi=600,facecolor='white')
fig.savefig(OUTPUT/'kitti_density.pdf',facecolor='white')
plt.close(fig)
with Image.open(png) as im:
    im.convert('RGB').save(OUTPUT/'kitti_density.tif',dpi=(600,600),compression='tiff_lzw')
(OUTPUT/'kitti_density_provenance.json').write_text(json.dumps({'input':str(SOURCE),'input_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'figure':str(png),'band':'plus/minus one sample standard deviation across training seeds 2026, 3407, 7351','test_pairs_per_run':555,'new_inference':False,'png_dpi':600},indent=2),encoding='utf-8')
print(str(png))

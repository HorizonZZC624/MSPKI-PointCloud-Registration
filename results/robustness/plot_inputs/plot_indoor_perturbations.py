from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(input_path, output_dir):
    with input_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'axes.linewidth': 1.1, 'xtick.major.width': 1, 'ytick.major.width': 1})
    output_dir.mkdir(parents=True, exist_ok=True)
    for condition, filename, xlabel in [('density', 'density_ir', 'Source points retained (%)'), ('noise', 'noise_ir', 'Gaussian coordinate noise std. (mm)')]:
        fig, ax = plt.subplots(figsize=(4, 3))
        for model, label, color in [('baseline', 'Baseline', '#1f77b4'), ('mspki', 'MSPKI', '#ff7f0e')]:
            selected = [r for r in rows if r['condition'] == condition and r['model'] == model]
            levels = [float(r['level']) for r in selected]
            values = [float(r['IR_percent']) for r in selected]
            ax.plot(range(len(levels)), values, 'o-', label=label, color=color, linewidth=2, markersize=6)
        ax.set_xticks(range(len(levels)), [f'{v:g}' for v in levels])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('IR (%)')
        ax.grid(True, linestyle=':', color='0.78', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right' if condition == 'noise' else 'lower left')
        fig.tight_layout(pad=0.4)
        for extension in ['png', 'tif', 'pdf']:
            extra = {'pil_kwargs': {'compression': 'tiff_lzw'}} if extension == 'tif' else {}
            fig.savefig(output_dir / f'{filename}.{extension}', dpi=600, bbox_inches='tight', pad_inches=0.03, **extra)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('indoor_perturbation_display.csv'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    draw(args.input, args.output_dir)


if __name__ == '__main__':
    main()

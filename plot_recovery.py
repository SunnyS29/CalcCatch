"""Render the five proposed ROI splits; requires optional matplotlib."""
import argparse
from pathlib import Path

import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/recovery'))
    args = parser.parse_args()
    baseline = np.load(args.results/'baseline.npz')
    candidate = np.load(args.results/'split-12-prominence-0.2.npz')
    before, after = baseline['labels'], candidate['labels']
    split_ids = [i for i in np.unique(before) if i > 0
                 and len(np.unique(after[before == i])) > 1]
    if not split_ids:
        raise SystemExit('No proposed splits to plot.')
    figure, axes = plt.subplots(1, len(split_ids), figsize=(3.5*len(split_ids), 4.2), squeeze=False)
    for index, (ax, roi_id) in enumerate(zip(axes[0], split_ids), 1):
        yy, xx = np.where(before == roi_id)
        y0, y1 = max(0, yy.min()-12), min(before.shape[0], yy.max()+13)
        x0, x1 = max(0, xx.min()-12), min(before.shape[1], xx.max()+13)
        ax.imshow(np.log1p(baseline['std'][y0:y1, x0:x1]), cmap='gray',
                  extent=(x0-.5, x1-.5, y1-.5, y0-.5), interpolation='nearest')
        ax.contour(np.arange(x0, x1), np.arange(y0, y1),
                   (before[y0:y1, x0:x1] == roi_id).astype(int),
                   levels=[.5], colors=['white'], linewidths=1)
        for child_id in np.unique(after[before == roi_id]):
            if child_id == 0:
                continue
            cy, cx = np.where(after == child_id)
            ax.scatter(cx.mean(), cy.mean(), s=70, facecolors='none',
                       edgecolors='#28d6eb', linewidths=1.8)
        landmarks = baseline['landmarks']
        nearby = landmarks[(landmarks[:, 0] >= x0) & (landmarks[:, 0] < x1)
                           & (landmarks[:, 1] >= y0) & (landmarks[:, 1] < y1)]
        ax.scatter(nearby[:, 0], nearby[:, 1], marker='+', s=90,
                   color='#ffb24a', linewidths=1.8)
        ax.set_title(f'Proposed split {index}')
        ax.set_xlim(x0-.5, x1-.5)
        ax.set_ylim(y1-.5, y0-.5)
        ax.set_xlabel('X (pixels)')
        if index == 1:
            ax.set_ylabel('Y (pixels)')
    handles = [Line2D([], [], marker='o', linestyle='', markerfacecolor='none',
                      markeredgecolor='#1494a5', label='Proposed child centroid'),
               Line2D([], [], marker='+', linestyle='', color='#ba6a00', label='Manual annotation')]
    figure.legend(handles=handles, loc='lower center', ncol=2, frameon=False)
    figure.suptitle('Exploratory splits: +4 matched annotations, +1 unmatched detection\n'
                    'White contour = original ROI; same recording used during exploration', fontsize=12)
    figure.tight_layout(rect=(0, .09, 1, .84))
    figure.savefig(args.results/'proposed-splits.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    main()

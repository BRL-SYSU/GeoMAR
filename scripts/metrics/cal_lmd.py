#!/usr/bin/env python3
"""Compute the 68-point landmark distance (LMD) between restored and GT faces.

LMD is reported as the normalized mean error (NME). The mean Euclidean
distance between corresponding landmarks is normalized by the GT inter-ocular
distance between landmarks 36 and 45. Lower values are better.

Example:
    python scripts/metrics/cal_lmd.py \
        --gt_dir ./datasets/celeba_512_validation \
        --methods GeoMAR:./results/GeoMAR_celeba_test_144/restored_faces \
        --out_dir ./results/metrics/lmd \
        --device cuda

Multiple methods can be evaluated in one run:
    --methods GeoMAR:/path/to/geomar DAEFR:/path/to/daefr
"""

import argparse
import csv
import os

import cv2
import numpy as np
from tqdm import tqdm


EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

# Suffixes removed when matching restored images to GT images.
STRIP_TOKENS = [
    '_00', '_0_', '_restored', '_result', '_out',
    '_output', '_fake', '_sr', '_hq', '_pred',
]


def normalize_stem(filename: str) -> str:
    """Convert a filename to a normalized image-pairing key."""
    stem = os.path.splitext(filename)[0]
    while True:
        old_stem = stem
        stem_lower = stem.lower()
        for token in STRIP_TOKENS:
            if stem_lower.endswith(token.lower()):
                stem = stem[:-len(token)]
                break
        if stem == old_stem:
            break

    stem = stem.strip('_-. ')
    stem = stem[-6:] if len(stem) > 6 else stem
    return stem.strip('_-. ').lower()


def build_index(directory: str) -> dict:
    """Build a mapping from normalized pairing keys to image paths."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory does not exist: {directory}')

    index = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(EXTENSIONS):
            continue

        key = normalize_stem(filename)
        if key in index:
            print(
                f'  [warning] Duplicate pairing key "{key}"; skipping '
                f'{filename} (already using {os.path.basename(index[key])})'
            )
            continue
        index[key] = os.path.join(directory, filename)
    return index


def pair_paths(restored_dir: str, gt_index: dict):
    """Pair restored images with GT images using normalized filename keys."""
    restored_index = build_index(restored_dir)
    pairs = []
    missing = []

    for key, gt_path in gt_index.items():
        if key in restored_index:
            pairs.append((restored_index[key], gt_path))
        else:
            missing.append(os.path.basename(gt_path))
    return pairs, missing


class LandmarkScorer:
    """Detect 68 facial landmarks with FAN and compute LMD/NME."""

    def __init__(self, device='cuda'):
        import face_alignment

        try:
            landmark_type = face_alignment.LandmarksType.TWO_D
        except AttributeError:
            landmark_type = face_alignment.LandmarksType._2D

        self.detector = face_alignment.FaceAlignment(
            landmark_type,
            device=device,
            flip_input=False,
        )

    def get_landmarks(self, image_path):
        """Return the first detected 68-point landmark set, or None."""
        image = cv2.imread(image_path)
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = self.detector.get_landmarks(image_rgb)
        if not predictions:
            return None
        return predictions[0]

    def calculate_lmd(self, restored_path, gt_path):
        """Return inter-ocular-normalized LMD, or None on detection failure."""
        restored_landmarks = self.get_landmarks(restored_path)
        gt_landmarks = self.get_landmarks(gt_path)
        if restored_landmarks is None or gt_landmarks is None:
            return None

        inter_ocular_distance = np.linalg.norm(
            gt_landmarks[36] - gt_landmarks[45]
        )
        if inter_ocular_distance <= 1e-8:
            return None

        point_distances = np.linalg.norm(
            restored_landmarks - gt_landmarks,
            axis=1,
        )
        return float(point_distances.mean() / inter_ocular_distance)


def evaluate_method(name, restored_dir, gt_index, scorer, out_dir):
    """Evaluate one restoration method and write its per-image LMD values."""
    print(f'\n=== Evaluating method: {name} ===')
    pairs, missing = pair_paths(restored_dir, gt_index)
    print(f'  Matched images: {len(pairs)}; missing restored images: {len(missing)}')
    if missing[:5]:
        suffix = ' ...' if len(missing) > 5 else ''
        print(f'  Missing examples: {missing[:5]}{suffix}')

    rows = []
    lmd_scores = []
    detection_failures = 0

    for restored_path, gt_path in tqdm(pairs, total=len(pairs), ncols=80):
        lmd = scorer.calculate_lmd(restored_path, gt_path)
        if lmd is None:
            detection_failures += 1
        else:
            lmd_scores.append(lmd)

        rows.append({
            'name': os.path.basename(gt_path),
            'restored': os.path.basename(restored_path),
            'lmd': lmd,
        })

    os.makedirs(out_dir, exist_ok=True)
    per_image_csv = os.path.join(out_dir, f'per_image_{name}.csv')
    with open(per_image_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['name', 'restored', 'lmd'])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'method': name,
        'n_pairs': len(pairs),
        'n_valid': len(lmd_scores),
        'LMD_NME': float(np.mean(lmd_scores)) if lmd_scores else None,
        'LmdFail_pct': (
            100.0 * detection_failures / len(pairs) if pairs else None
        ),
    }

    print(f'  Mean LMD: {summary["LMD_NME"]}')
    print(f'  Landmark detection failure rate: {summary["LmdFail_pct"]}%')
    print(f'  Per-image results saved to: {per_image_csv}')
    return summary


def parse_methods(method_arguments):
    """Parse method specifications in NAME:DIRECTORY format."""
    methods = []
    for item in method_arguments:
        if ':' not in item:
            raise ValueError(
                f'Expected NAME:DIRECTORY for --methods, received: {item}'
            )
        name, path = item.split(':', 1)
        methods.append((name.strip(), path.strip()))
    return methods


def main():
    parser = argparse.ArgumentParser(
        description='Calculate LMD/NME for restored face images.'
    )
    parser.add_argument('--gt_dir', required=True, help='GT image directory.')
    parser.add_argument(
        '--methods',
        required=True,
        nargs='+',
        help='Methods in NAME:RESTORED_DIRECTORY format.',
    )
    parser.add_argument(
        '--out_dir',
        default='./metric_out',
        help='Output directory.',
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Landmark detector device: cuda or cpu.',
    )
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    print(f'GT directory: {args.gt_dir}')
    print(f'Methods: {[method[0] for method in methods]}')

    print('\nLoading the facial landmark detector...')
    scorer = LandmarkScorer(device=args.device)

    print('\nBuilding the GT index...')
    gt_index = build_index(args.gt_dir)
    print(f'  Number of GT images: {len(gt_index)}')

    summaries = [
        evaluate_method(name, path, gt_index, scorer, args.out_dir)
        for name, path in methods
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, 'summary_lmd.csv')
    fields = ['method', 'n_pairs', 'n_valid', 'LMD_NME', 'LmdFail_pct']
    with open(summary_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print('\n' + '=' * 66)
    print('LMD summary (lower is better)')
    print('=' * 66)
    print(f'{"Method":<20}{"Pairs":>10}{"Valid":>10}{"LMD/NME":>14}')
    print('-' * 66)
    for summary in summaries:
        lmd_text = (
            f'{summary["LMD_NME"]:.6f}'
            if summary['LMD_NME'] is not None
            else 'N/A'
        )
        print(
            f'{summary["method"]:<20}'
            f'{summary["n_pairs"]:>10}'
            f'{summary["n_valid"]:>10}'
            f'{lmd_text:>14}'
        )
    print('=' * 66)
    print(f'Summary saved to: {summary_csv}')


if __name__ == '__main__':
    main()

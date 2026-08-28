import os
import argparse
import torch
import tqdm
from PIL import Image

import pyiqa


def calculate_niqe_folder():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Path to image folder')
    parser.add_argument('--save_name', type=str, default='niqe.txt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pyiqa NIQE
    metric = pyiqa.create_metric('niqe', device=device)
    metric.eval()

    img_names = sorted(os.listdir(args.folder))
    scores = []

    with torch.no_grad():
        for name in tqdm.tqdm(img_names):
            if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(args.folder, name)
            img = Image.open(img_path).convert('RGB')

            score = metric(img)
            scores.append(score.item())

    mean_score = sum(scores) / len(scores)

    with open(args.save_name, 'w') as f:
        f.write(f'NIQE: {mean_score}\n')

    print('NIQE:', mean_score)


if __name__ == '__main__':
    calculate_niqe_folder()

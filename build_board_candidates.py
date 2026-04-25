"""
Erzeugt Kandidaten-Templates fuer Board-Karten aus allen 52 Screenshots.
Speichert nur in separatem Kandidaten-Ordner.
"""
import csv
import os
import cv2
from detectors.card_detector import CardDetector
from utils.config import TABLE_LAYOUTS


def main() -> None:
    det = CardDetector()
    det._apply_layout_profile('acipayam_heads_up')

    folder = 'C:/poker-1/Neuer Ordner'
    layout = TABLE_LAYOUTS['acipayam_heads_up']
    ref_w, ref_h = layout['reference_size']

    out_dir = 'assets/board_card_templates_candidates'
    os.makedirs(out_dir, exist_ok=True)

    report_path = 'board_candidates_report.csv'

    min_score = 0.90
    max_gap = 0.02

    rows = []
    saved = 0

    for i in range(1, 53):
        fname = f'Screenshot_{i}.png'
        img = cv2.imread(f'{folder}/{fname}')
        if img is None:
            continue

        h, w = img.shape[:2]
        sx = w / ref_w
        sy = h / ref_h

        for bi, roi in enumerate(layout['community_cards']):
            rx = int(roi[0] * sx)
            ry = int(roi[1] * sy)
            rw = int(roi[2] * sx)
            rh = int(roi[3] * sy)
            crop = img[ry:ry + rh, rx:rx + rw]
            if crop is None or crop.size == 0:
                continue

            has_surface = det._has_community_card_surface(crop)
            if not has_surface:
                rows.append({
                    'screenshot': i,
                    'position': bi,
                    'status': 'no_card_surface',
                    'best_name': '',
                    'best_score': '',
                    'second_name': '',
                    'second_score': '',
                    'gap': '',
                    'candidate_saved': 'no',
                    'candidate_file': '',
                })
                continue

            surf = det._extract_card_surface(crop)
            if surf is None or surf.size == 0:
                surf = crop

            gray = cv2.cvtColor(surf, cv2.COLOR_BGR2GRAY)

            scores = {}
            for card_name, tmpl_list in det.board_card_templates.items():
                if isinstance(tmpl_list, list):
                    best_for_label = max(
                        det._score_card_template(gray, t) for t in tmpl_list
                    )
                else:
                    best_for_label = det._score_card_template(gray, tmpl_list)
                scores[card_name] = float(best_for_label)

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_name, best_score = ranked[0]
            second_name, second_score = (
                ranked[1] if len(ranked) > 1 else ("", -1.0)
            )
            gap = best_score - second_score

            candidate = bool(best_score >= min_score and gap <= max_gap)
            out_name = (
                f'{best_name}__SS{i:02d}_board{bi}_'
                f's{best_score:.3f}_g{gap:.3f}.png'
            )

            if candidate:
                cv2.imwrite(os.path.join(out_dir, out_name), surf)
                saved += 1

            rows.append({
                'screenshot': i,
                'position': bi,
                'status': 'card_surface',
                'best_name': best_name,
                'best_score': f'{best_score:.3f}',
                'second_name': second_name,
                'second_score': f'{second_score:.3f}',
                'gap': f'{gap:.3f}',
                'candidate_saved': 'yes' if candidate else 'no',
                'candidate_file': out_name if candidate else '',
            })

    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'screenshot',
                'position',
                'status',
                'best_name',
                'best_score',
                'second_name',
                'second_score',
                'gap',
                'candidate_saved',
                'candidate_file',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'Report: {report_path}')
    print(f'Candidates saved: {saved} -> {out_dir}')


if __name__ == '__main__':
    main()

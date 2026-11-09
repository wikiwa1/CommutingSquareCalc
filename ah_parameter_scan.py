"""Numerical level-6 screen for the S(1,1,2,2) parameter family.

At this index the classified alternatives are TLJ (dimension 132 at
level 6) and Asaeda-Haagerup (dimension 133). Floating-point dimensions
are numerical evidence, not rank certificates. No exhaustive search
or list of exceptional parameters is asserted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import sympy as sp
from pathalg import Level
from ah_demo import ah_square


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', nargs='+', default=['0', 'pi/2', 'pi', '3*pi/2', 'pi/3'])
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--block', type=int, default=160, help='initial eigenspace search size (must exceed 133)')
    parser.add_argument('--append', action='store_true', help='preserve previous attempts in the output file')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('ah_parameter_scan_results.json'))
    args = parser.parse_args()
    if args.block <= 133 or args.seconds <= 0:
        parser.error('--block must exceed 133 and --seconds must be positive')
    results = json.loads(args.output.read_text(encoding='utf-8'))['results'] if args.append and args.output.exists() else []
    for label in args.parameters:
        s = sp.sympify(label)
        if s.free_symbols or s.is_real is not True:
            parser.error('Parameters must be explicit real expressions.')
        start = time.monotonic()
        print(f's={s}: computing level 6', flush=True)
        row = {'s': str(s), 's_numeric': float(s), 'n': 6, 'backend': 'numpy', 'method': 'iterative', 'initial_block': args.block}
        def progress(info):
            if info['round'] % 5 == 0:
                print(f"  round {info['round']}: {info['found']} directions, next cosine {info['next']:.9f}, residual {info['residual']:.2e}, {time.monotonic()-start:.1f}s", flush=True)
            if time.monotonic() - start > args.seconds:
                raise TimeoutError('Time allowance exceeded; no dimension reported.')
        try:
            lv = Level(ah_square(s, backend='numpy'), 6, backend='numpy')
            dim = lv.relative_commutant_dimension((0, 1), (6, 0), method='iterative', block=args.block, progress=progress)
            info = lv._last_intersection
            row.update(dimension=dim, numerical_classification={132: 'TLJ-compatible', 133: 'AH-compatible'}.get(dim, 'unexpected: investigate'),
                       gap=list(lv._last_angle_gap), residual=info.get('residual'), iterations=info.get('iterations'))
            print(f"  dimension={dim}; {row['numerical_classification']}; gap={row['gap']}", flush=True)
            del lv
        except (TimeoutError, ArithmeticError, RuntimeError) as exc:
            row['unresolved'] = str(exc)
            print(f'  unresolved: {exc}', flush=True)
        row['seconds'] = time.monotonic() - start
        results.append(row)
        args.output.write_text(json.dumps({'level': 6, 'results': results, 'exhaustive': False}, indent=2), encoding='utf-8')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()

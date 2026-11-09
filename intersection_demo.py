"""Intersections of actual members of two one-parameter connection families.

Run python intersection_demo.py [--section hadamard|ah] [--max-n N].
Part A uses H(r)=f4_family(r), not diagonal modifications of F_4.
Reference: Bakshi--Guin, arXiv:2212.07197v3, Section 7. The Fourier-conjugate
examples of arXiv:2510.14499 are a different experiment.

The circle parameters are a=exp(i*s), b=exp(i*t), theta=exp(i*(t-s)).
The library's adjoint convention conjugates theta; orders and cos^2 agree.
Inclusion norms, finite centers and mixing from C_1 are finite-level data,
not automatic proofs of limiting index or factoriality. JSON preserves
every pair and its full-precision numerical results.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import sympy as sp
import ah_demo
from pathalg import InsufficientMemory
from pathalg.examples import f4_family, hadamard
from pathalg.intersection import LevelPair, analyse

BACKEND = "numpy"  # double precision: "numpy" or "cupy"
HADAMARD_MAX_N = 5
AH_MAX_N = 4
CHECK_TOL = 5e-7
OUTPUT_JSON = Path(__file__).with_name("intersection_demo_results.json")

HADAMARD_PAIRS = [
    ("H01", 0, 0, "identical Fourier inputs"),
    ("H02", sp.pi/7, sp.pi/7, "identical non-Fourier inputs"),
    ("H03", sp.pi/7, 8*sp.pi/7, "pi shift: different matrices, same algebra"),
    ("H04", 0, sp.pi/2, "order 4 phase: commuting pair, index 8"),
    ("H05", sp.pi/7, 9*sp.pi/14, "shifted order 4 pair"),
    ("H06", 0, sp.pi/3, "order 6 phase: noncommuting, index 12"),
    ("H07", 0, 2*sp.pi/3, "order 3 phase: squared-phase order 3, index 12"),
    ("H08", sp.pi/7, 10*sp.pi/21, "shifted order 6 pair"),
    ("H09", 0, sp.pi/4, "order 8 phase: index 16, delayed stabilization"),
    ("H10", 0, sp.pi/5, "order 10 phase: index 20, delayed stabilization"),
    ("H11", 0, 1, "irrational relative phase"),
    ("H12", sp.pi/7, sp.pi/7+1, "shifted irrational pair"),
    ("H13", 0, sp.sqrt(2), "second irrational relative phase"),
    ("H14", 0, sp.pi/3+sp.Rational(1,100), "irrational perturbation of order 6"),
]
S = sp.pi/3
S_PRIMES = [sp.pi/3, sp.pi/2, sp.Rational(7,5), 2*sp.pi/3, sp.pi,
            4*sp.pi/3, 5*sp.pi/3, sp.Rational(1,10)]


def phase_data(s, t):
    """Classify exactly, never infer rationality by rounding a floating point.

    For (t-s)/pi=p/q reduced, q=ord(theta^2), and the index benchmark is 4q.
    Odd ord(theta) is handled by theta -> -theta (same embedded subfactor),
    which has order 2q; using 2*ord(theta) would give the wrong answer.
    """
    delta=sp.simplify(sp.sympify(t)-sp.sympify(s))
    ratio=sp.simplify(delta/sp.pi)
    info=dict(delta=str(delta),delta_latex=sp.latex(delta),ratio_pi=str(ratio),
              phase_order=None,squared_phase_order=None,index_benchmark=None)
    if ratio.is_Rational:
        p,q=int(ratio.p),int(ratio.q)
        info.update(phase_order=2*q//math.gcd(abs(p),2*q),squared_phase_order=q,
                    index_benchmark=4*q,category="equal" if q==1 else "finite",
                    commuting_expected=q<=2)
    else:
        # SymPy leaves (pi/3 + 1/100)/pi undecided. A rational multiple
        # of pi plus a nonzero algebraic remainder has an irrational ratio
        # to pi, since otherwise pi would be algebraic.
        coefficient=sp.expand(delta).coeff(sp.pi)
        remainder=sp.simplify(delta-coefficient*sp.pi)
        irrational=(ratio.is_irrational is True or
                    (coefficient.is_Rational and remainder.is_algebraic is True
                     and remainder.is_zero is False))
        info.update(category="infinite" if irrational else "unclassified",
                    commuting_expected=False if irrational else None)
    return info


def result_record(result):
    """Save data without retaining large operator stacks between pairs."""
    rows=[]
    for r in result.records:
        v=r.spectrum
        rows.append(dict(n=r.p,dim_ambient=v.dim,dim_intersection=v.dim_intersection,
            interior_count=len(v.distinct),interior_eigenvalues=[[float(x),int(m)] for x,m in v.distinct],
            top_cos2=v.top,inclusion_norm2=r.index_estimate,center_dim=r.algebra.centre_dim,
            block_sizes=sorted(int(x) for x in r.algebra.sizes),mix_from_1=r.dobrushin))
    return dict(records=rows,components=[dict(levels=[p+1 for p in c.levels],
                trace=c.trace,starts=c.starts) for c in result.components])


def validate(data, phase=None, same=False, individual_index=4):
    """Level-1 analytic formula, equality controls and internal consistency.

    Not reaching a limiting index at finite depth is not a failure.
    """
    checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
    rows=data["records"]
    check("spectral and algebra dimensions agree",all(sum(b*b for b in r["block_sizes"])==r["dim_intersection"] for r in rows))
    for a,b in zip(rows,rows[1:]):
        check(f"spectrum at n={a['n']} persists at n={b['n']}",
              all(any(abs(x-y)<CHECK_TOL for y,_ in b["interior_eigenvalues"]) for x,_ in a["interior_eigenvalues"]))
    if phase is not None:
        same=phase["category"]=="equal"
        if not same:
            check("level-1 intersection dimension is 3",rows[0]["dim_intersection"]==3)
            value=float(sp.N(sp.cos(sp.sympify(phase["delta"]))**2))
            actual=rows[0]["interior_eigenvalues"]
            check("level-1 spectrum matches cos^2(t-s)",not actual if value<CHECK_TOL else
                  len(actual)==1 and abs(actual[0][0]-value)<CHECK_TOL)
        q=phase["squared_phase_order"]
        if q is not None and q>1:
            expected=[math.cos(k*math.pi/q)**2 for k in range(1,q)]
            check("spectrum lies in rational-phase spectral set",all(any(abs(x-y)<CHECK_TOL for y in expected)
                  for r in rows for x,_ in r["interior_eigenvalues"]))
            check("inclusion norms do not exceed index benchmark",all(r["inclusion_norm2"]<=4*q+CHECK_TOL for r in rows))
        if phase["commuting_expected"]:
            check("commuting control has no interior spectrum",all(r["interior_count"]==0 for r in rows))
    if same:
        check("equal pair has full dimension",all(r["dim_intersection"]==r["dim_ambient"] for r in rows))
        check("equal pair has no interior spectrum",all(r["interior_count"]==0 for r in rows))
        check("equal pair norms equal individual index",all(abs(r["inclusion_norm2"]-individual_index)<CHECK_TOL for r in rows))
    return checks


def table(data):
    print("     n  dim B_n  dim C_n  eig(0,1)  max cos^2   incl norm^2  center  mix from 1")
    for r in data["records"]:
        top="-" if r["top_cos2"] is None else f"{r['top_cos2']:.6f}"
        mix="-" if r["mix_from_1"] is None else f"{r['mix_from_1']:.4f}"
        print(f"    {r['n']:2}  {r['dim_ambient']:7}  {r['dim_intersection']:7}  {r['interior_count']:8}  "
              f"{top:>9}  {r['inclusion_norm2']:12.4f}  {r['center_dim']:6}  {mix:>10}")
    persistent=[c for c in data["components"] if len(c["levels"])==len(data["records"])]
    if len(persistent)>1:
        print(f"    {len(persistent)} components persist through computed levels only:")
        for c in persistent:
            print(f"      trace={c['trace']:.6f}; starting vertices={c['starts']}")


def run_pair(cid,family,s,t,note,square_s,square_t,depth):
    print(f"\n  {cid}: (s,t)=({s}, {t}) -- {note}",flush=True)
    phase=phase_data(s,t) if family=="hadamard" else None
    if phase:
        print(f"    phase classification: {phase['category']}; ord(theta)={phase['phase_order']}; "
              f"ord(theta^2)={phase['squared_phase_order']}; index benchmark={phase['index_benchmark']}")
    data=dict(id=cid,family=family,s=str(s),t=str(t),s_latex=sp.latex(s),t_latex=sp.latex(t),
              note=note,max_n=depth,phase=phase)
    start=time.time()
    try:
        data.update(result_record(analyse(LevelPair(square_s,square_t,depth,backend=BACKEND))))
    except (InsufficientMemory,ArithmeticError) as exc:
        data.update(status="unresolved",error=str(exc),seconds=time.time()-start)
        print(f"    UNRESOLVED: {exc}")
        return data
    checks=validate(data,phase,same=sp.simplify(t-s)==0,
                    individual_index=float(sp.N(ah_demo.DELTA**2)) if family=="ah" else 4)
    data.update(checks=checks,status="ok" if all(c["passed"] for c in checks) else "failed",seconds=time.time()-start)
    table(data)
    if phase and phase["category"]=="finite":
        first=next((r["n"] for r in data["records"] if abs(r["inclusion_norm2"]-phase["index_benchmark"])<CHECK_TOL),None)
        print(f"    Index benchmark first reached at n={first}." if first else
              "    Index benchmark not reached at these depths; no contradiction.")
    print(f"    Checks: {sum(c['passed'] for c in checks)}/{len(checks)} passed; {data['seconds']:.1f}s")
    for c in checks:
        if not c["passed"]:
            print(f"      FAILED: {c['name']}")
    return data


def compare_pairs(a,b):
    if a["status"]!="ok" or b["status"]!="ok" or len(a["records"])!=len(b["records"]):
        return False
    for x,y in zip(a["records"],b["records"]):
        if any(x[k]!=y[k] for k in ("dim_intersection","center_dim","block_sizes","interior_count")):
            return False
        if abs(x["inclusion_norm2"]-y["inclusion_norm2"])>CHECK_TOL:
            return False
        if any(abs(v-w)>CHECK_TOL or m!=n for (v,m),(w,n) in zip(x["interior_eigenvalues"],y["interior_eigenvalues"])):
            return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section",choices=["all","hadamard","ah"],default="all")
    parser.add_argument("--max-n",type=int,help="override the default depth of the selected sections")
    parser.add_argument("--json",type=Path,default=OUTPUT_JSON)
    args=parser.parse_args()
    if args.max_n is not None and args.max_n<1:
        parser.error("--max-n must be positive")
    payload=dict(schema_version=1,backend=BACKEND,
        versions=dict(python=platform.python_version(),numpy=np.__version__,sympy=sp.__version__),
        spectral_tolerance=1e-9,cluster_tolerance=1e-8,check_tolerance=CHECK_TOL,pairs=[],comparisons=[])
    def save():
        args.json.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print("C_n = A_(n,0)^(s) intersect A_(n,0)^(t). Dimensions are complex vector-space dimensions.")
    print("eig(0,1) counts DISTINCT interior eigenvalues; max cos^2 is not an angle in radians.")
    print("A dash means no interior spectrum, or no transition yet. Inclusion norms are finite-level data.")
    print("center and mix from 1 do not by themselves establish limiting factoriality.")
    if args.section in ("all","hadamard"):
        depth=args.max_n or HADAMARD_MAX_N
        print(f"\nA. H(r)=f4_family(r), q=i*exp(i*r); levels 1..{depth}")
        print("   Reference: Bakshi--Guin, arXiv:2212.07197v3, Section 7.")
        print("   This replaces the diagonal Fourier-conjugate examples of arXiv:2510.14499.")
        for cid,s,t,note in HADAMARD_PAIRS:
            s,t=sp.sympify(s),sp.sympify(t)
            # Equality bases are largest; four levels suffice for these controls.
            n=min(depth,4) if args.max_n is None and phase_data(s,t)["category"]=="equal" else depth
            payload["pairs"].append(run_pair(cid,"hadamard",s,t,note,
                hadamard(f4_family(s),backend=BACKEND),hadamard(f4_family(t),backend=BACKEND),n))
            save()
        by_id={p["id"]:p for p in payload["pairs"]}
        for left,right in [("H04","H05"),("H06","H07"),("H06","H08"),("H11","H12")]:
            passed=compare_pairs(by_id[left],by_id[right])
            payload["comparisons"].append(dict(pairs=[left,right],passed=passed))
            print(f"\n  Relative-phase comparison {left}/{right}: {'PASS' if passed else 'FAILED'}")
    if args.section in ("all","ah"):
        depth=args.max_n or AH_MAX_N
        print(f"\nB. S(1,1,2,2), s=pi/3; levels 1..{depth}; individual index={float(sp.N(ah_demo.DELTA**2)):.6f}")
        square_s=ah_demo.ah_square(S,backend=BACKEND)
        for i,t in enumerate(S_PRIMES,1):
            payload["pairs"].append(run_pair(f"B{i:02}","ah",S,t,
                "same-parameter control" if t==S else "unequal AH parameters",
                square_s,ah_demo.ah_square(t,backend=BACKEND),depth))
            save()
    payload["complete"]=all(p["status"]=="ok" for p in payload["pairs"]) and all(c["passed"] for c in payload["comparisons"])
    save()
    print(f"\nSaved {len(payload['pairs'])} pairs to {args.json.name}; complete={payload['complete']}.")
    if not payload["complete"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()

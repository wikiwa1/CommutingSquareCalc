"""Generate the mathematical exposition from the recorded default run.

Edit intersection_report_template.tex for prose; tables come from the JSON.
The resulting output/pdf/intersection_results_explained.tex is standalone.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Code/intersection_demo_results.json"
TEMPLATE = ROOT / "Code/intersection_report_template.tex"
TARGET = ROOT / "output/pdf/intersection_results_explained.tex"


def number(value, digits=6):
    return "---" if value is None else f"{value:.{digits}f}"


def result_table(pairs, caption, label, omit_top=False, ambient=False):
    def row(r):
        values = [str(r["n"])]
        if ambient:
            values.append(str(r["dim_ambient"]))
        return values + [
            str(r["dim_intersection"]), str(r["interior_count"]),
            r"$\ast$" if omit_top else number(r["top_cos2"]),
            number(r["inclusion_norm2"], 4)]

    rows = [row(r) for r in pairs[0]["records"]]
    for pair in pairs[1:]:
        assert [row(r) for r in pair["records"]] == rows, pair["id"]
    headings = [r"$n$"] + ([r"$b_n$"] if ambient else []) + [
        r"$c_n$", r"$k_n$", r"$\lambda_n$", r"$\ell_n$"]
    return "\n".join([
        r"\begin{center}\begin{minipage}{\linewidth}\centering",
        r"\small\begin{tabular}{" + "r" * len(headings) + r"}\toprule",
        " & ".join(headings) + r"\\\midrule",
        *[" & ".join(r) + r"\\" for r in rows],
        r"\bottomrule\end{tabular}",
        r"\captionof{table}{" + caption + r"}\label{" + label + "}",
        r"\end{minipage}\end{center}"])


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert data.get("complete"), "A complete successful run is required."
    pairs = {p["id"]: p for p in data["pairs"]}
    assert set(pairs) == {f"H{i:02}" for i in range(1, 15)} | {f"B{i:02}" for i in range(1, 9)}
    for ident, pair in pairs.items():
        depth = 4 if ident.startswith("B") or ident in {"H01", "H02", "H03"} else 5
        assert [r["n"] for r in pair["records"]] == list(range(1, depth + 1))
        assert all(c["passed"] for c in pair["checks"])
    assert len(data["comparisons"]) == 4 and all(c["passed"] for c in data["comparisons"])
    replacements = {}

    def table(key, ids, caption, **kw):
        replacements[key] = result_table([pairs[i] for i in ids], caption, "tab:" + key.lower(), **kw)

    table("EQUAL", ["H01", "H02", "H03"], r"The three equal-subfactor pairs. Here $b_n=4^n$.")
    table("COMMUTING", ["H04", "H05"], r"The two pairs with $t-s=\pi/2$.")
    table("ORDER3", ["H06", "H07", "H08"], r"The three pairs with $q=3$.")
    table("ORDER4", ["H09"], r"The pair $(s,t)=(0,\pi/4)$, for which $q=4$.")
    table("ORDER5", ["H10"], r"The pair $(s,t)=(0,\pi/5)$, for which $q=5$.")
    table("IRR1", ["H11", "H12"], r"The two pairs with $t-s=1$.")
    table("IRR2", ["H13"], r"The pair $(s,t)=(0,\sqrt2)$.")
    table("PERTURB", ["H14"], r"The pair $(s,t)=(0,\pi/3+1/100)$.")
    table("AH_EQUAL", ["B01"], r"The equal-parameter pair in the $S(1,1,2,2)$ family.", ambient=True)
    table("AH_PROPER", [f"B{i:02}" for i in range(2, 9)],
          r"Common data for the seven unequal pairs in Appendix~\ref{sec:ah}. "
          r"The asterisk denotes the parameter-dependent value in Table~\ref{tab:ahmax}.",
          omit_top=True, ambient=True)

    lines = [r"\begin{center}\begin{minipage}{\linewidth}\centering\small",
             r"\begin{tabular}{cllrr}\toprule",
             r"Label & $(s,t)$ & $\Delta=t-s$ & $q$ & $I(s,t)$\\\midrule"]
    for i in range(1, 15):
        p = pairs[f"H{i:02}"]
        phase = p["phase"]
        q = str(phase["squared_phase_order"]) if phase["squared_phase_order"] else r"\infty"
        index = str(phase["index_benchmark"]) if phase["index_benchmark"] else r"\infty"
        lines.append(p["id"] + r" & $(" + p["s_latex"] + "," + p["t_latex"] +
                     r")$ & $" + phase["delta_latex"] + r"$ & $" + q + r"$ & $" + index + r"$\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\captionof{table}{All Hadamard pairs. The last column is the theoretical index, not a finite-level measurement.}\label{tab:parameters}",
              r"\end{minipage}\end{center}"]
    replacements["PARAMETERS"] = "\n".join(lines)

    lines = [r"\begin{center}\begin{minipage}{\linewidth}\centering\small",
             r"\begin{tabular}{clrrrr}\toprule",
             r"Label & $t$ & $\lambda_1$ & $\lambda_2$ & $\lambda_3$ & $\lambda_4$\\\midrule"]
    for i in range(1, 9):
        p = pairs[f"B{i:02}"]
        lines.append(p["id"] + " & $" + p["t_latex"] + "$ & " +
                     " & ".join(number(r["top_cos2"]) for r in p["records"]) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\captionof{table}{Every pair in the $S(1,1,2,2)$ calculation, with $s=\pi/3$.}\label{tab:ahmax}",
              r"\end{minipage}\end{center}"]
    replacements["AH_MAXIMA"] = "\n".join(lines)

    groups = [
        (["H01", "H02", "H03"], "H01--H03"),
        (["H04", "H05"], "H04--H05"),
        (["H06", "H07", "H08"], "H06--H08"),
        ([f"H{i:02}" for i in range(9, 15)], "H09--H14"),
        (["B01"], "B01"), ([f"B{i:02}" for i in range(2, 9)], "B02--B08")]
    lines = [r"\begin{center}\begin{minipage}{\linewidth}\centering\small",
             r"\begin{tabular}{lll}\toprule",
             r"Pairs & $(z_n)_{n=1}^{N}$ & $(d_{1,n})_{n=2}^{N}$\\\midrule"]
    for ids, name in groups:
        def entries(p):
            return (", ".join(str(r["center_dim"]) for r in p["records"]),
                    ", ".join(number(r["mix_from_1"], 4) for r in p["records"][1:]))
        values = entries(pairs[ids[0]])
        assert all(entries(pairs[i]) == values for i in ids)
        lines.append(name + " & $(" + values[0] + ")$ & $(" + values[1] + r")$\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\captionof{table}{Center dimensions and trace-transition coefficients. Here $N$ is the last computed level for the indicated pairs.}\label{tab:centers}",
              r"\end{minipage}\end{center}"]
    replacements["CENTERS"] = "\n".join(lines)

    replacements["CHECKS"] = str(sum(len(p["checks"]) for p in pairs.values()))
    replacements["ROWS"] = str(sum(len(p["records"]) for p in pairs.values()))
    result = TEMPLATE.read_text(encoding="utf-8")
    for key, value in replacements.items():
        token = "@@" + key + "@@"
        assert token in result, key
        result = result.replace(token, value)
    assert "@@" not in result, "Unexpanded template token"
    TARGET.write_text(result, encoding="utf-8")
    print(f"Wrote {TARGET}: {len(pairs)} pairs, {replacements['ROWS']} recorded levels.")


if __name__ == "__main__":
    main()

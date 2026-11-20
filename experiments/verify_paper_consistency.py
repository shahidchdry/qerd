"""
Automated consistency verifier for `paper.tex` against all saved JSON artifacts,
IBM Runtime Job-ID ledger, figures, cross-references, and citations.

Run from repository root:
  python experiments/verify_paper_consistency.py
"""
from __future__ import annotations

import json
import os
import re


def main() -> None:
    tex = open("paper.tex", encoding="utf-8").read()
    ledger = json.load(open("analysis/ibm_kingston_job_ledger.json", encoding="utf-8"))
    m_sg = json.load(open("projects/sp500_qubo_and_spinglass/artifacts/metrics.json", encoding="utf-8"))["heavy_hex_spinglass"]

    errors = []

    # 1. Verify EVERY Job ID appearing in paper.tex exists verbatim in analysis/ibm_kingston_job_ledger.json
    all_valid_jids = set()
    for entry in ledger.values():
        for k, v in entry.items():
            if isinstance(v, str) and len(v) == 20 and v.startswith("db"):
                all_valid_jids.add(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and len(x) == 20 and x.startswith("db"):
                        all_valid_jids.add(x)

    tex_jids = set(re.findall(r"\bdb[a-z0-9]{18}\b", tex))
    missing_jids = tex_jids - all_valid_jids
    if missing_jids:
        errors.append(f"Job IDs in paper.tex not in ledger: {missing_jids}")

    # 2. Verify NO stale or inconsistent patterns remain anywhere in paper.tex
    stale_patterns = [
        (r"85\.3\s*\\pm\s*2\.3", "2-instance SD 85.3±2.3"),
        (r"56\.0\s*\\pm", "2-instance SD 56.0±"),
        (r"21\.1\\%", "Old inconsistent no-zz Gap Closed 21.1%"),
        (r"p\{=\}4\s+QAOA|p=4\s+QAOA", "Old p=4 QAOA label"),
        (r"heterogeneous hardware-adapted", "Retracted heterogeneous couplers claim"),
        (r"multi-valley sampling", "Overstated multi-valley claim"),
        (r"\+7\.7", "Unmatched +7.7 pp oracle claim"),
        (r"\+8\.2", "Unmatched +8.2 pp oracle claim"),
        (r"O_a\(s'\)", "Old Eq. 12 s' typo"),
        (r"circumvents generic", "Old Omega(P) circumvent overclaim"),
        (r"\\input\{", "External \\input macro dependency"),
    ]
    for pat, desc in stale_patterns:
        if re.search(pat, tex):
            errors.append(f"Stale pattern found ({desc}): {pat}")

    # 3. Verify unified seed=42 Gap Closed arithmetic against E_rand = -0.3125
    e_r = m_sg["untrained_baseline"]["mean_energy"]
    e_sa = m_sg["sa_ceiling"]["mean_energy"]
    expected_gaps = {
        "coord_psr": round(100.0 * (e_r - m_sg["coord_psr_qpu"]["mean_energy"]) / (e_r - e_sa), 1),
        "untrained_qpu": round(100.0 * (e_r - m_sg["untrained_qpu"]["mean_energy"]) / (e_r - e_sa), 1),
        "nozz": round(100.0 * (e_r - m_sg["nozz_qpu"]["mean_energy"]) / (e_r - e_sa), 1),
        "spsa": round(100.0 * (e_r - m_sg["spsa_qpu"]["mean_energy"]) / (e_r - e_sa), 1),
        "qerd": round(100.0 * (e_r - m_sg["qerd_qpu"]["mean_energy"]) / (e_r - e_sa), 1),
    }
    for name, val in expected_gaps.items():
        s_val = f"+{val:.1f}\\%" if name == "untrained_qpu" else f"{val:.1f}\\%"
        if s_val not in tex:
            errors.append(f"Expected unified seed=42 Gap Closed {name}={s_val} not found in paper.tex")

    # 4. Verify all citations in \cite{...} exist in \bibitem{...}
    cited = set()
    for grp in re.findall(r"\\cite\{([^}]+)\}", tex):
        for c in grp.split(","):
            cited.add(c.strip())
    defined_bib = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    undef_cites = cited - defined_bib
    if undef_cites:
        errors.append(f"Undefined citations: {undef_cites}")

    # 5. Verify all \ref{...} / \eqref{...} exist in \label{...}
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex))
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    undef_refs = refs - labels
    if undef_refs:
        errors.append(f"Undefined refs: {undef_refs}")

    # 6. Verify all \includegraphics files exist on disk
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    missing_figs = [f for f in figs if not os.path.exists(f)]
    if missing_figs:
        errors.append(f"Missing figure files: {missing_figs}")

    print(
        f"Checked: {len(tex_jids)} Job IDs, {len(cited)} citations, {len(refs)} refs, "
        f"{len(figs)} figures, {len(stale_patterns)} stale-claim rules, 5 unified Gap-Closed values."
    )
    print(f"Expected unified seed=42 Gap Closed: {expected_gaps}")
    if errors:
        print("FAILURES:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)
    print("ALL CHECKS PASSED WITH ZERO ERRORS!")


if __name__ == "__main__":
    main()

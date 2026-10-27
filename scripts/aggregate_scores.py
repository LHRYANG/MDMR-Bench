"""Aggregate per-dimension scores from {sample_id}_results.json beside generated images.

Per-case weighted dimension score:
  first 2 questions weight 0.25 each (50% total),
  remaining questions share 50% equally.
  Edge cases: 1 question -> that score; 2 questions -> simple mean.

Directory-level score = arithmetic mean of per-case weighted dimension scores
over all cases under that directory.

Aggregation levels: from each case's parent directory up to --top (inclusive).
Outputs are saved under --benchmark-root/analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CHECKLIST_KEYS = ("following", "consistency", "composition", "minimality", "quality")
DIM_WEIGHTS = {
    "following": 0.20,
    "consistency": 0.25,
    "composition": 0.25,
    "minimality": 0.20,
    "quality": 0.10,
}
SCORE_KEYS = (*CHECKLIST_KEYS, "overall")
TASK_NAMES = ("fusion", "style_transfer", "text_logo", "multiple_extraction", "reasoning")
FLAT_TASKS = {"multiple_extraction"}  # tasks whose cases are direct children of the task dir


def weighted_score(xs: list[float]) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return 0.5 * xs[0] + 0.5 * xs[1]
    rest = xs[2:]
    return 0.25 * xs[0] + 0.25 * xs[1] + 0.5 * (sum(rest) / len(rest))


def to_answer_list(v: Any) -> list[float]:
    if not isinstance(v, list):
        return []
    out: list[float] = []
    for x in v:
        if isinstance(x, str):
            s = x.strip().upper()
            if s == "Y":
                out.append(1.0)
            elif s == "N":
                out.append(0.0)
    return out


def case_scores(result_obj: dict[str, Any]) -> dict[str, float | None]:
    answer_dict = result_obj.get("Answer Dict") or {}
    out: dict[str, float | None] = {}
    for k in CHECKLIST_KEYS:
        out[k] = weighted_score(to_answer_list(answer_dict.get(k)))

    present = [(k, out[k]) for k in CHECKLIST_KEYS if out[k] is not None]
    if not present:
        out["overall"] = None
    else:
        wsum = sum(DIM_WEIGHTS[k] for k, _ in present)
        out["overall"] = sum(v * DIM_WEIGHTS[k] for k, v in present) / wsum
    return out


def ancestors_up_to(case_dir: Path, top: Path) -> list[Path]:
    try:
        rel_parts = case_dir.relative_to(top).parts
    except ValueError:
        return []
    result: list[Path] = []
    cur = case_dir.parent
    for _ in range(len(rel_parts)):
        result.append(cur)
        if cur == top:
            break
        cur = cur.parent
    return result


def _avg_score_dicts(dicts: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Mean over a list of {key: float} dicts, ignoring None."""
    out: dict[str, float | None] = {}
    for k in SCORE_KEYS:
        vals = [d[k] for d in dicts if d.get(k) is not None]
        out[k] = mean(vals) if vals else None
    return out


def compute_balanced_summary(
    cases: list[dict[str, Any]],
    benchmark_root: Path,
) -> dict[str, Any]:
    """Hierarchical averaging: cases -> subtask -> task -> benchmark.

    For tasks in FLAT_TASKS (e.g. multiple_extraction), skip the subtask layer
    and average cases directly.
    """
    by_task_sub: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    flat_task_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        try:
            rel = Path(c["case_dir"]).relative_to(benchmark_root)
        except ValueError:
            continue
        if not rel.parts:
            continue
        task = c.get("task_id") or rel.parts[0]
        if task in FLAT_TASKS:
            flat_task_cases[task].append(c)
            continue
        subtask = c.get("subtask_id") or (rel.parts[1] if len(rel.parts) >= 3 else "(root)")
        by_task_sub[task][subtask].append(c)

    task_entries: dict[str, dict[str, Any]] = {}
    task_scores: list[dict[str, float | None]] = []

    for task in TASK_NAMES:
        if task in FLAT_TASKS:
            task_cases = flat_task_cases.get(task, [])
            if not task_cases:
                continue
            score = _avg_score_dicts([c["scores"] for c in task_cases])
            task_entries[task] = {
                "score": score,
                "n_cases": len(task_cases),
                "subtasks": None,
            }
        else:
            sub_map = by_task_sub.get(task, {})
            if not sub_map:
                continue
            subtasks: dict[str, dict[str, Any]] = {}
            sub_scores: list[dict[str, float | None]] = []
            for sub_name in sorted(sub_map.keys()):
                sub_cases = sub_map[sub_name]
                sub_score = _avg_score_dicts([c["scores"] for c in sub_cases])
                subtasks[sub_name] = {"score": sub_score, "n_cases": len(sub_cases)}
                sub_scores.append(sub_score)
            score = _avg_score_dicts(sub_scores)
            task_entries[task] = {
                "score": score,
                "n_cases": sum(len(v) for v in sub_map.values()),
                "subtasks": subtasks,
            }
        task_scores.append(task_entries[task]["score"])

    benchmark_score = _avg_score_dicts(task_scores) if task_scores else {k: None for k in SCORE_KEYS}
    return {
        "benchmark": {
            "score": benchmark_score,
            "n_cases": len(cases),
            "n_tasks_used": len(task_entries),
        },
        "tasks": task_entries,
    }


def _fmt_score(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "-"


def _score_row(rel_path: str, level: str, n_cases: int, score: dict[str, float | None]) -> list[Any]:
    row = [rel_path, level, n_cases]
    for k in SCORE_KEYS:
        v = score.get(k)
        row.append(f"{v:.4f}" if v is not None else "")
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="Generated image and scoring directory for one run, e.g. outputs/flux2",
    )
    p.add_argument(
        "--top",
        type=Path,
        default=None,
        help="Aggregate up to (and including) this directory. Default: --benchmark-root.",
    )
    args = p.parse_args()

    benchmark_root = args.benchmark_root.resolve()
    top = args.top.resolve() if args.top is not None else benchmark_root
    out_dir = benchmark_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path for path in benchmark_root.rglob("*_results.json")
        if path.name == f"{path.parent.name}_results.json"
    )
    cases: list[dict[str, Any]] = []
    skipped = 0
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        if not isinstance(obj.get("Answer Dict"), dict):
            skipped += 1
            continue
        scores = case_scores(obj)
        cases.append({
            "sample_id": obj.get("sample_id"),
            "task_id": obj.get("task_id"),
            "subtask_id": obj.get("subtask_id"),
            "case_dir": str(f.parent.resolve()),
            "result_json": str(f.resolve()),
            "scores": scores,
        })

    per_case_path = out_dir / "per_case.jsonl"
    with per_case_path.open("w", encoding="utf-8") as fh:
        for c in cases:
            row = {k: c[k] for k in ("sample_id", "task_id", "subtask_id", "case_dir", "result_json")}
            for k in SCORE_KEYS:
                row[k] = c["scores"][k]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_dir: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    for c in cases:
        case_dir = Path(c["case_dir"])
        for anc in ancestors_up_to(case_dir, top):
            by_dir[str(anc)].append(c["scores"])

    rows: list[dict[str, Any]] = []
    for d, score_list in by_dir.items():
        dp = Path(d)
        try:
            rel = dp.relative_to(top.parent)
        except ValueError:
            rel = dp
        row: dict[str, Any] = {
            "directory": d,
            "rel_path": str(rel),
            "depth": len(rel.parts),
            "n_cases": len(score_list),
        }
        for k in SCORE_KEYS:
            vals = [s[k] for s in score_list if s.get(k) is not None]
            row[k] = mean(vals) if vals else None
        rows.append(row)

    rows.sort(key=lambda r: (r["depth"], r["rel_path"]))

    (out_dir / "by_directory.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cols = ["rel_path", "depth", "n_cases", *SCORE_KEYS]
    csv_path = out_dir / "by_directory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["rel_path"],
                r["depth"],
                r["n_cases"],
                *(f"{r[c]:.4f}" if r[c] is not None else "" for c in cols[3:]),
            ])

    print(f"benchmark_root={benchmark_root}  cases={len(cases)}  skipped={skipped}")
    short = {"following": "fol", "consistency": "con", "composition": "com", "minimality": "min", "quality": "qua"}
    for r in rows[:20]:
        print(f"  [d{r['depth']}] n={r['n_cases']:>4}  {r['rel_path']}")
        parts = []
        for k in CHECKLIST_KEYS:
            v = r[k]
            parts.append(f"{short[k]}={v:.4f}" if v is not None else f"{short[k]}=-")
        ov = r["overall"]
        ov_s = f"{ov:.4f}" if ov is not None else "-"
        print("      " + "  ".join(parts) + f"  | overall={ov_s}")
    print(f"outputs:\n  {per_case_path}\n  {out_dir / 'by_directory.json'}\n  {csv_path}")

    balanced = compute_balanced_summary(cases, benchmark_root)
    balanced_json = out_dir / "balanced_summary.json"
    balanced_json.write_text(json.dumps(balanced, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    balanced_csv = out_dir / "balanced_summary.csv"
    cols2 = ["rel_path", "level", "n_cases", *SCORE_KEYS]
    with balanced_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols2)
        w.writerow(_score_row("benchmark", "benchmark", balanced["benchmark"]["n_cases"], balanced["benchmark"]["score"]))
        for task in TASK_NAMES:
            task_entry = balanced["tasks"].get(task)
            if not task_entry:
                continue
            w.writerow(_score_row(task, "task", task_entry["n_cases"], task_entry["score"]))
            if task_entry["subtasks"]:
                for sub_name, sub_entry in task_entry["subtasks"].items():
                    w.writerow(_score_row(f"{task}/{sub_name}", "subtask", sub_entry["n_cases"], sub_entry["score"]))

    print("\n--- balanced (subtask-avg -> task-avg -> benchmark-avg) ---")

    def _print(label: str, n: int, s: dict[str, float | None], indent: int = 0):
        pad = "  " * indent
        print(f"{pad}{label}  n={n}")
        parts = [f"{short[k]}={_fmt_score(s.get(k))}" for k in CHECKLIST_KEYS]
        print(f"{pad}  " + "  ".join(parts) + f"  | overall={_fmt_score(s.get('overall'))}")

    _print("[benchmark]", balanced["benchmark"]["n_cases"], balanced["benchmark"]["score"])
    for task in TASK_NAMES:
        task_entry = balanced["tasks"].get(task)
        if not task_entry:
            continue
        _print(f"[{task}]", task_entry["n_cases"], task_entry["score"], indent=1)
        if task_entry["subtasks"]:
            for sub_name, sub_entry in task_entry["subtasks"].items():
                _print(f"[{sub_name}]", sub_entry["n_cases"], sub_entry["score"], indent=2)
    print(f"\nbalanced outputs:\n  {balanced_json}\n  {balanced_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

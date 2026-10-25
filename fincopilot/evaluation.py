"""Evaluate a model on the held-out and real-world test sets.

    # In Colab, after training: fine-tuned model vs the same base model without the adapter
    python -m fincopilot.evaluation --backend hf --adapter outputs/fincopilot-lora --compare-base

    # Locally with Ollama: run each model, then merge the two reports
    python -m fincopilot.evaluation --backend ollama --model llama3.1:8b --prompt-style full --name base
    python -m fincopilot.evaluation --backend ollama --model fincopilot --name finetuned
    python -m fincopilot.evaluation --compare reports/base.json reports/finetuned.json

Metrics (each test item is a single model reply, with no self-repair, so this is the
model's raw ability):
  answer accuracy   calculation questions where the model's call produces exactly the right result
                    (right tool + values; equivalent forms such as 72 months vs 6 years count)
  routing accuracy  right tool chosen
  argument match    extracted values identical to the label after validation
  valid call rate   output parsed as a tool call and passed the Pydantic form
  asked when info was missing / avoided tools for non-calculation questions
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from .grounding import ungrounded_numbers
from .parsing import ParseError, ToolCall, parse_model_output
from .prompts import with_system
from .tools import ToolDomainError, clean_numbers, get_tool, to_json

DEFAULT_SETS = ["data/test.jsonl", "data/realworld_test.jsonl"]


# --- scoring ----------------------------------------------------------------------------------

def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=0.006)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    return False


def normalized_args(name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tool = get_tool(name)
    try:
        return clean_numbers(tool.validate(params).model_dump(exclude_none=True)) if tool else None
    except ValidationError:
        return None


def outcome(name: str, params: Dict[str, Any]) -> Tuple[str, Any]:
    tool = get_tool(name)
    if tool is None:
        return "unknown_tool", None
    try:
        return "ok", tool.run(params).result
    except ValidationError:
        return "invalid", None
    except ToolDomainError as exc:
        return "domain", str(exc)


def score(item: Dict[str, Any], raw: str) -> Dict[str, Any]:
    expected = item["expected"]
    r: Dict[str, Any] = {"id": item["id"], "set": item.get("_set", "test"), "type": item.get("type"),
                         "question": item["messages"][-1]["content"], "expected": expected, "raw": raw,
                         "valid_json": True, "predicted_action": None, "predicted_tool": None,
                         "args_valid": False, "args_exact": False, "answer_correct": False, "correct": False}
    try:
        parsed = parse_model_output(raw)
    except ParseError as exc:
        r.update(valid_json=False, predicted_action="invalid", error=str(exc))
        return r
    if isinstance(parsed, ToolCall):
        r.update(predicted_action="tool", predicted_tool=parsed.name, predicted_parameters=parsed.parameters)
    else:
        r["predicted_action"] = "text"

    if expected["action"] == "tool":
        if r["predicted_action"] != "tool":
            return r
        status, value = outcome(parsed.name, parsed.parameters)
        r["args_valid"] = status in ("ok", "domain")
        if parsed.name == expected["name"]:
            r["args_exact"] = _close(normalized_args(parsed.name, parsed.parameters),
                                     normalized_args(expected["name"], expected["parameters"]))
            exp_status, exp_value = outcome(expected["name"], expected["parameters"])
            r["answer_correct"] = status == exp_status and _close(value, exp_value)
        r["correct"] = r["answer_correct"]
    else:  # clarify or text: the right move is a plain-text reply
        r["correct"] = r["predicted_action"] == "text"
        r["asked_question"] = r["predicted_action"] == "text" and "?" in raw
    return r


# --- running ------------------------------------------------------------------------------------

def load_sets(paths: Sequence[str], limit: Optional[int] = None, seed: int = 0) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        name = "realworld" if "realworld" in os.path.basename(path) else "test"
        if limit and len(rows) > limit and name != "realworld":
            rng = random.Random(seed)
            by_type: Dict[str, List[Dict]] = defaultdict(list)
            for row in rows:
                by_type[row["type"]].append(row)
            picked = []
            for rows_of_type in by_type.values():  # stratified, so every type stays represented
                k = max(1, round(limit * len(rows_of_type) / len(rows)))
                picked += rng.sample(rows_of_type, min(k, len(rows_of_type)))
            rows = picked
        for row in rows:
            row["_set"] = name
        items += rows
    return items


def run(backend, items: Sequence[Dict[str, Any]], prompt_style: str = "compact", batch_size: int = 16,
        explain: bool = False, verbose: bool = True) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    start = time.time()
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        prompts = [with_system(item["messages"], prompt_style) for item in chunk]
        t0 = time.time()
        outputs = backend.generate_batch(prompts, max_new_tokens=256)
        per_item = (time.time() - t0) / len(chunk)
        for item, raw in zip(chunk, outputs):
            res = score(item, raw)
            res["seconds"] = round(per_item, 3)
            results.append(res)
        if verbose:
            done = len(results)
            acc = sum(r["correct"] for r in results) / done
            print(f"  {done}/{len(items)}  running accuracy {acc:.1%}  ({time.time() - start:.0f}s)", flush=True)
    if explain:
        check_explanations(backend, items, results, prompt_style, batch_size)
    return results


def check_explanations(backend, items, results, prompt_style: str, batch_size: int) -> None:
    """For correctly answered questions, ask the model to explain the tool result and grounding-check it."""
    todo = [(item, res) for item, res in zip(items, results) if res["answer_correct"]]
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        prompts = []
        for item, res in chunk:
            exp = item["expected"]
            result = get_tool(exp["name"]).run(exp["parameters"]).result
            msgs = list(item["messages"]) + [{"role": "assistant", "content": to_json({"name": exp["name"], "parameters": exp["parameters"]})},
                                             {"role": "tool", "content": to_json(result)}]
            prompts.append(with_system(msgs, prompt_style))
        outputs = backend.generate_batch(prompts, max_new_tokens=256)
        for (item, res), text in zip(chunk, outputs):
            exp = item["expected"]
            tool = get_tool(exp["name"])
            run_ = tool.run(exp["parameters"])
            bad = ungrounded_numbers(text, tool.allowed_numbers(run_.args, run_.result))
            res["explanation"] = text.strip()
            res["explanation_grounded"] = bool(text.strip()) and not bad
            res["explanation_ungrounded"] = bad


# --- summaries ------------------------------------------------------------------------------------

def _rate(values: Sequence[bool]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        tool_rows = [r for r in rows if r["expected"]["action"] == "tool"]
        clarify = [r for r in rows if r["expected"]["action"] == "clarify"]
        text = [r for r in rows if r["expected"]["action"] == "text"]
        non_tool = clarify + text
        out = {
            "items": len(rows),
            "overall_accuracy": _rate([r["correct"] for r in rows]),
            "calculation_questions": len(tool_rows),
            "answer_accuracy": _rate([r["answer_correct"] for r in tool_rows]),
            "routing_accuracy": _rate([r["predicted_tool"] == r["expected"]["name"] for r in tool_rows]),
            "argument_match": _rate([r["args_exact"] for r in tool_rows]),
            "valid_call_rate": _rate([r["args_valid"] for r in tool_rows]),
            "missing_info_questions": len(clarify),
            "asked_when_info_missing": _rate([r["correct"] for r in clarify]),
            "non_calculation_questions": len(text),
            "no_tool_when_not_needed": _rate([r["correct"] for r in text]),
            "false_tool_call_rate": _rate([r["predicted_action"] == "tool" for r in non_tool]),
            "unparseable_rate": _rate([not r["valid_json"] for r in rows]),
            "avg_seconds_per_item": round(sum(r.get("seconds", 0) for r in rows) / max(len(rows), 1), 3),
        }
        explained = [r for r in rows if "explanation_grounded" in r]
        if explained:
            out["explanation_grounded_rate"] = _rate([r["explanation_grounded"] for r in explained])
        return out

    summary = {"all": block(results)}
    for name in sorted({r["set"] for r in results}):
        summary[name] = block([r for r in results if r["set"] == name])

    per_tool: Dict[str, Dict[str, Any]] = {}
    for name in sorted({r["expected"]["name"] for r in results if r["expected"]["action"] == "tool"}):
        rows = [r for r in results if r["expected"]["action"] == "tool" and r["expected"]["name"] == name]
        per_tool[name] = {"n": len(rows), "answer_accuracy": _rate([r["answer_correct"] for r in rows]),
                          "routing_accuracy": _rate([r["predicted_tool"] == name for r in rows])}
    summary["per_tool"] = per_tool
    confusions = Counter((r["expected"]["name"], r["predicted_tool"] or r["predicted_action"])
                         for r in results if r["expected"]["action"] == "tool" and r["predicted_tool"] != r["expected"]["name"])
    summary["routing_mistakes"] = [{"expected": e, "predicted": p, "count": c} for (e, p), c in confusions.most_common(10)]
    return summary


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


ROWS = [("answer_accuracy", "Answer accuracy (calculation questions)"), ("routing_accuracy", "Routing accuracy (right tool)"),
        ("argument_match", "Argument match (exact values)"), ("valid_call_rate", "Valid tool calls"),
        ("asked_when_info_missing", "Asked when info was missing"), ("no_tool_when_not_needed", "No tool for concept/advice/off-topic"),
        ("false_tool_call_rate", "False tool-call rate (lower is better)"), ("overall_accuracy", "Overall accuracy (all items)"),
        ("explanation_grounded_rate", "Explanations passing the grounding check")]


def markdown_report(runs: Dict[str, Dict[str, Any]]) -> str:
    """runs: name -> {"summary": ..., "results": [...], "meta": {...}}"""
    names = list(runs)
    lines = ["# Financial Copilot evaluation", ""]
    for name in names:
        meta = runs[name].get("meta", {})
        lines.append(f"- **{name}**: {meta.get('description', '')}")
    lines.append("")
    for section, title in (("all", "All test items"), ("test", "Held-out test set (phrasings never seen in training)"),
                           ("realworld", "Hand-written real-world questions")):
        if not all(section in runs[n]["summary"] for n in names):
            continue
        n_items = runs[names[0]]["summary"][section]["items"]
        lines += [f"## {title} ({n_items} items)", "", "| Metric | " + " | ".join(names) + " |",
                  "|---|" + "---|" * len(names)]
        for key, label in ROWS:
            values = [runs[n]["summary"][section].get(key) for n in names]
            if all(v is None for v in values):
                continue
            lines.append(f"| {label} | " + " | ".join(_pct(v) for v in values) + " |")
        lines.append("")

    last = names[-1]
    per_tool = runs[last]["summary"]["per_tool"]
    lines += [f"## Per-tool answer accuracy ({last})", "", "| Tool | n | Answer accuracy | Routing |", "|---|---|---|---|"]
    for tool, s in sorted(per_tool.items(), key=lambda kv: (kv[1]["answer_accuracy"] or 0, kv[0])):
        lines.append(f"| {tool} | {s['n']} | {_pct(s['answer_accuracy'])} | {_pct(s['routing_accuracy'])} |")
    mistakes = runs[last]["summary"]["routing_mistakes"]
    if mistakes:
        lines += ["", f"## Most common routing mistakes ({last})", "", "| Expected | Predicted | Count |", "|---|---|---|"]
        lines += [f"| {m['expected']} | {m['predicted']} | {m['count']} |" for m in mistakes]
    failures = [r for r in runs[last]["results"] if not r["correct"]][:15]
    if failures:
        lines += ["", f"## Sample failures ({last})", ""]
        for r in failures:
            exp = r["expected"]
            want = f"{exp['name']} {json.dumps(exp.get('parameters', {}))}" if exp["action"] == "tool" else exp["action"]
            lines += [f"- **Q:** {r['question']}", f"  - expected: `{want}`",
                      f"  - got: `{r['raw'].strip()[:300]}`"]
    return "\n".join(lines) + "\n"


def save_run(out_dir: str, name: str, results, summary, meta) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "results": results}, f, indent=1)
    return path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["hf", "ollama", "openai"])
    p.add_argument("--model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--data", nargs="+", default=DEFAULT_SETS)
    p.add_argument("--limit", type=int, default=None, help="sample N items from the held-out test set (real-world set always runs in full)")
    p.add_argument("--prompt-style", choices=["compact", "full"], default="compact")
    p.add_argument("--compare-base", action="store_true", help="hf only: also evaluate the base model (adapter off, full prompt)")
    p.add_argument("--explain", action="store_true", help="also grounding-check the model's explanations")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="reports")
    p.add_argument("--compare", nargs="+", help="merge saved report JSONs into one markdown report")
    args = p.parse_args(argv)

    if args.compare:
        runs = {}
        for path in args.compare:
            with open(path, encoding="utf-8") as f:
                runs[os.path.splitext(os.path.basename(path))[0]] = json.load(f)
        report = markdown_report(runs)
        out = os.path.join(args.out, "comparison.md")
        os.makedirs(args.out, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        print(f"Saved {out}")
        return
    if not args.backend:
        p.error("--backend is required (or use --compare)")

    from .backends import make_backend
    items = load_sets(args.data, args.limit)
    print(f"Evaluating {len(items)} items")
    backend = make_backend(args.backend, model=args.model, adapter=args.adapter, host=args.host)
    runs: Dict[str, Dict[str, Any]] = {}

    if args.compare_base:
        if not hasattr(backend, "base_model"):
            p.error("--compare-base needs --backend hf")
        print("== Base model (adapter disabled, full prompt) ==")
        with backend.base_model():
            results = run(backend, items, "full", args.batch_size, args.explain)
        meta = {"description": f"{args.model or 'Llama 3.1 8B Instruct'} without fine-tuning, full tool-description prompt"}
        runs["base"] = {"results": results, "summary": summarize(results), "meta": meta}
        save_run(args.out, "base", results, runs["base"]["summary"], meta)

    name = args.name or ("finetuned" if args.adapter or args.compare_base else args.backend)
    print(f"== {name} ==")
    results = run(backend, items, args.prompt_style, args.batch_size, args.explain)
    meta = {"description": f"backend={args.backend} model={args.model} adapter={args.adapter} prompt={args.prompt_style}"}
    runs[name] = {"results": results, "summary": summarize(results), "meta": meta}
    save_run(args.out, name, results, runs[name]["summary"], meta)

    report = markdown_report(runs)
    out = os.path.join(args.out, "report.md" if len(runs) == 1 else "comparison.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

"""Function-level audit of two branches against their merge base.

Answers "which branch is actually ahead?" for every Python function, using git
history only. Each function is compared in canonical form (ast.unparse with
docstrings removed), so comments and formatting never count as differences.

For a function that differs between A and B:
  B-ahead    A still has the merge-base version (or an older B version); B changed it
  A-ahead    the mirror case
  diverged   both sides changed it independently since the merge base
  B-new      added on B after the split
  B-removed  existed at the merge base, B deleted it (A still has it)
The commit that introduced each side's current version is reported, so every
claim can be checked with `git show <commit>`.

Usage (from the repository root):
    python tools/branch_audit.py main pilot_trial > audit.md
"""
import argparse
import ast
import subprocess
from collections import Counter
from functools import lru_cache

# Terms that mark a difference as tied to one environment's layout or config.
LAYOUT_TERMS = (
    "school", "shift", "department", "dept", "hospital", "ilc", "patient_bed",
    "served_beds", "foot_traffic", "location_tag", "intraday", "weekday",
    "item_database", "sku_database", "recovery", "ward", "is_cluttered",
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


@lru_cache(maxsize=None)
def file_at(ref: str, path: str):
    try:
        return git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return None


def _strip_docstring(node):
    body = getattr(node, "body", None)
    if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        node.body = body[1:] or [ast.Pass()]
    for child in ast.iter_child_nodes(node):
        _strip_docstring(child)


@lru_cache(maxsize=None)
def functions(ref: str, path: str):
    """Qualified name -> canonical source; None if the file is absent or unparsable."""
    src = file_at(ref, path)
    if src is None:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    _strip_docstring(tree)
    out, rest = {}, []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.unparse(node)
        elif isinstance(node, ast.ClassDef):
            body_rest = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = ast.unparse(sub)
                else:
                    body_rest.append(ast.unparse(sub))
            out[f"{node.name}.<fields>"] = "\n".join(body_rest)
        else:
            rest.append(ast.unparse(node))
    out["<module>"] = "\n".join(rest)
    return out


def history(base: str, ref: str, path: str):
    """Commits on ref since base that touched path, oldest first."""
    log = git("log", "--reverse", "--date-order", "--format=%H", f"{base}..{ref}", "--", path)
    return log.split()


def introduced_by(base: str, ref: str, path: str, name: str):
    """First commit (since base) at which `name` took its current form on ref."""
    current = (functions(ref, path) or {}).get(name)
    found = None
    prev = (functions(base, path) or {}).get(name)
    for commit in history(base, ref, path):
        version = (functions(commit, path) or {}).get(name)
        if version != prev and version == current:
            found = commit
        prev = version
    return found


def older_versions(base: str, ref: str, path: str, name: str):
    return {(functions(c, path) or {}).get(name) for c in history(base, ref, path)}


def describe(commit):
    if not commit:
        return ""
    return git("log", "-1", "--format=%h %ad %s", "--date=short", commit).strip()


def classify(base, a, b, path, name):
    fa, fb, f0 = (functions(r, path) or {} for r in (a, b, base))
    va, vb, v0 = fa.get(name), fb.get(name), f0.get(name)
    if va == vb:
        return None
    if va is None:
        return ("A-removed" if v0 is not None else "B-new"), None, introduced_by(base, b, path, name)
    if vb is None:
        return ("B-removed" if v0 is not None else "A-new"), introduced_by(base, a, path, name), None
    if va == v0 or va in older_versions(base, b, path, name):
        return "B-ahead", None, introduced_by(base, b, path, name)
    if vb == v0 or vb in older_versions(base, a, path, name):
        return "A-ahead", introduced_by(base, a, path, name), None
    return "diverged", introduced_by(base, a, path, name), introduced_by(base, b, path, name)


def layout_flag(a, b, path, name):
    text = ((functions(a, path) or {}).get(name, "") + (functions(b, path) or {}).get(name, "")).lower()
    hits = sorted({t for t in LAYOUT_TERMS if t in text})
    return ", ".join(hits)


def collect(base, a, b):
    """Return (rows, file_only): differing functions and files present on one side only."""
    paths = sorted({p for ref in (a, b) for p in git("ls-tree", "-r", "--name-only", ref).split()
                    if p.endswith(".py")})
    rows, file_only = [], []
    for path in paths:
        fa, fb = functions(a, path), functions(b, path)
        if fa is None or fb is None:
            side = a if fb is None else b
            file_only.append((path, side, file_at(base, path) is not None))
            continue
        for name in sorted(set(fa) | set(fb)):
            result = classify(base, a, b, path, name)
            if result:
                status, ca, cb = result
                rows.append((path, name, status, describe(ca), describe(cb), layout_flag(a, b, path, name)))
    return rows, file_only


def print_report(base, a, b, rows, file_only):
    print(f"# Branch audit: A = `{a}`, B = `{b}`\n")
    print(f"Merge base: `{describe(base)}`\n")
    counts = Counter(r[2] for r in rows)
    print("| Status | Functions |\n| --- | --- |")
    for status, n in sorted(counts.items()):
        print(f"| {status} | {n} |")
    print("\n## Files present on one side only\n")
    print("| File | Only on | Existed at merge base |\n| --- | --- | --- |")
    for path, side, at_base in file_only:
        print(f"| `{path}` | {side} | {'yes' if at_base else 'no'} |")
    print("\n## Differing functions\n")
    current = None
    for path, name, status, ca, cb, flag in rows:
        if path != current:
            print(f"\n### `{path}`\n")
            print(f"| Function | Status | Changed on {a} by | Changed on {b} by | Layout terms |")
            print("| --- | --- | --- | --- | --- |")
            current = path
        print(f"| `{name}` | {status} | {ca} | {cb} | {flag} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()
    base = git("merge-base", args.a, args.b).strip()
    rows, file_only = collect(base, args.a, args.b)
    print_report(base, args.a, args.b, rows, file_only)


if __name__ == "__main__":
    main()

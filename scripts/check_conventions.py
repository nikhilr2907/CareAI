import ast
import sys
from pathlib import Path


def check_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        docstring = node.body[0]
        if isinstance(node, ast.Module):
            errors.append(f"{path}:{docstring.lineno} remove the module docstring")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if docstring.end_lineno != docstring.lineno:
                errors.append(f"{path}:{docstring.lineno} keep method docstrings on one line")
    return errors


def main() -> int:
    errors = [error for filename in sys.argv[1:] for error in check_file(Path(filename))]
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

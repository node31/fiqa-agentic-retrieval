"""Count code lines the way the spec's "~2,000 lines" budget means it.

    python scripts/count_loc.py

Counts src/ excluding the data loader (src/beir_data.py), as the rule says, and splits each file into
logical code lines vs docstring/comment/blank lines. This repo is deliberately documentation-heavy
(it is also a learning resource), so both numbers are reported.
"""
import ast
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"beir_data.py"}


def split_counts(path: Path) -> tuple[int, int]:
    src = path.read_text(encoding="utf-8")
    doc_lines: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    code_lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                            tokenize.ENDMARKER) and tok.start[0] not in doc_lines:
            code_lines.update(range(tok.start[0], tok.end[0] + 1))
    total = len(src.splitlines())
    return len(code_lines), total


def main():
    rows, code_sum, total_sum = [], 0, 0
    for p in sorted((ROOT / "src").rglob("*.py")):
        if p.name in EXCLUDE:
            continue
        code, total = split_counts(p)
        rows.append((str(p.relative_to(ROOT)), code, total))
        code_sum, total_sum = code_sum + code, total_sum + total
    for name, code, total in rows:
        print(f"{name:40s} code {code:5d}   total {total:5d}")
    print(f"{'TOTAL src/ (excl. data loader)':40s} code {code_sum:5d}   total {total_sum:5d} "
          f"({100 * (1 - code_sum / total_sum):.0f}% docs/comments/blank)")


if __name__ == "__main__":
    main()

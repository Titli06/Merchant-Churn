"""Build the analytical warehouse by running sql/*.sql in filename order.

Usage
-----
    python src/warehouse.py                 # build everything
    python src/warehouse.py --only 04       # re-run one step
    python src/warehouse.py --query "SELECT ..."

The DuckDB file lands at data/warehouse.duckdb. Nothing in the SQL layer is
DuckDB-specific except sql/00_schema.sql -- see that file's header.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
DB_PATH = ROOT / "data" / "warehouse.duckdb"


def split_statements(sql: str) -> list[str]:
    """Split on semicolons that are not inside a string literal or line comment."""
    out, buf, in_str, in_comment = [], [], False, False
    i = 0
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if in_comment:
            buf.append(ch)
            if ch == "\n":
                in_comment = False
        elif in_str:
            buf.append(ch)
            if ch == "'":
                if nxt == "'":
                    buf.append(nxt)
                    i += 1
                else:
                    in_str = False
        elif ch == "-" and nxt == "-":
            in_comment = True
            buf.append(ch)
        elif ch == "'":
            in_str = True
            buf.append(ch)
        elif ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute(f"SET file_search_path='{ROOT}'")
    return con


def build(only: str | None = None, verbose: bool = True) -> None:
    files = sorted(SQL_DIR.glob("*.sql"))
    if only:
        files = [f for f in files if f.name.startswith(only)]
        if not files:
            raise SystemExit(f"No sql file starting with {only!r}")
        # 00_schema always has to run first: the views it defines are session
        # objects over parquet, and later scripts depend on them.
        schema = SQL_DIR / "00_schema.sql"
        if schema not in files:
            files = [schema] + files

    con = connect()
    try:
        for path in files:
            statements = split_statements(path.read_text())
            t0 = time.perf_counter()
            for stmt in statements:
                if not re.sub(r"--.*", "", stmt).strip():
                    continue
                try:
                    con.execute(stmt)
                except Exception as exc:  # noqa: BLE001
                    head = stmt.strip().splitlines()[0][:110]
                    raise RuntimeError(f"{path.name} failed on: {head}\n  {exc}") from None
            if verbose:
                print(f"  {path.name:<40} {len(statements):>2} stmt  {time.perf_counter()-t0:6.2f}s")
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only sql files with this prefix")
    ap.add_argument("--query", default=None, help="run an ad-hoc query and print it")
    args = ap.parse_args()

    if args.query:
        con = connect(read_only=True)
        print(con.execute(args.query).df().to_string(index=False))
        con.close()
        return

    print(f"Building warehouse -> {DB_PATH.relative_to(ROOT)}")
    build(only=args.only)
    con = connect(read_only=True)
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY 1"
    ).df()["table_name"].tolist()
    print("\nTables built:")
    for t in tables:
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"  {t:<40} {n:>12,} rows")
    con.close()


if __name__ == "__main__":
    main()

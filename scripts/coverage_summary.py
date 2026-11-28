"""Render pytest-cov JSON totals for GitHub and the README."""

import argparse
import json
from pathlib import Path

START = "<!-- project-coverage:start -->"
END = "<!-- project-coverage:end -->"


def summary(report):
    totals = report["totals"]
    lines, statements = totals["covered_lines"], totals["num_statements"]
    branches, possible = totals["covered_branches"], totals["num_branches"]
    return (
        "| Scope | Covered | Total | Coverage |\n"
        "| --- | ---: | ---: | ---: |\n"
        f"| Executable lines | {lines} | {statements} | {lines/statements:.2%} |\n"
        f"| Branches | {branches} | {possible} | {branches/possible:.2%} |\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args(argv)
    table = summary(json.loads(args.report.read_text(encoding="utf-8")))
    if args.readme is None:
        print("### ALIGN project coverage\n\n" + table)
        return
    original = args.readme.read_text(encoding="utf-8")
    before, after_start = original.split(START)
    _, after = after_start.split(END)
    args.readme.write_text(f"{before}{START}\n\n{table}\n{END}{after}", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

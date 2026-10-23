#!/usr/bin/env python3
"""RunPod preflight and resumable orchestration for the complete study."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from drone_study.config import PROJECT_ROOT


def memory_gib() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return pages * page_size / 1024**3


def preflight(output_dir: Path, workers: int, args: argparse.Namespace) -> dict:
    cpu_count = os.cpu_count() or 1
    memory = memory_gib()
    output_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_dir)
    checks = {
        "python": sys.version.split()[0],
        "logical_cpus": cpu_count,
        "memory_gib": round(memory, 2) if memory is not None else "unavailable",
        "free_storage_gib": round(disk.free / 1024**3, 2),
        "workers": workers,
        "mode": args.mode,
        "replications": args.replications,
        "cv_folds": args.cv_folds if args.mode == "extended-paper" else None,
        "candidates_per_family": (
            args.search_candidates if args.mode == "extended-paper" else None
        ),
        "estimated_model_fits": (
            args.replications * (3 * args.search_candidates * args.cv_folds + 4)
            if args.mode == "extended-paper"
            else args.replications * 3
        ),
        "output_dir": str(output_dir),
    }
    if workers > cpu_count:
        raise ValueError(
            f"workers ({workers}) exceed logical CPUs ({cpu_count}); reduce --workers"
        )
    if memory is not None and memory < 8:
        raise ValueError("less than 8 GiB RAM detected; use a larger RunPod instance")
    if disk.free < 5 * 1024**3:
        raise ValueError("less than 5 GiB free storage detected")
    return checks


def run_checked(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def safe_fresh_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError("--fresh may delete only a child directory inside this project")
    if resolved.exists():
        shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight and execute the standard or prolonged RunPod study."
    )
    parser.add_argument("--mode", choices=("standard", "extended-paper"), default="extended-paper")
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path,
                        default=Path("results/extended"))
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--search-candidates", type=int, default=12)
    parser.add_argument("--workers", "--threads", dest="workers", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.fresh and args.resume:
        parser.error("--fresh and --resume are mutually exclusive")
    if not 1 <= args.replications <= 30:
        parser.error("--replications must be between 1 and 30")
    if not 1 <= args.workers <= args.replications:
        parser.error("--workers must be between 1 and --replications")
    if not 2 <= args.cv_folds <= 210:
        parser.error("--cv-folds must be between 2 and 210")
    if not 1 <= args.search_candidates <= 12:
        parser.error("--search-candidates must be between 1 and 12")
    if args.bootstrap_resamples < 1000:
        parser.error("--bootstrap-resamples must be at least 1000")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if args.fresh:
        safe_fresh_output(output_dir)
    plan = preflight(output_dir, args.workers, args)
    print(json.dumps(plan, indent=2), flush=True)

    python = sys.executable
    if not args.skip_tests:
        run_checked([python, "validate_pipeline.py"])
        run_checked([python, "-m", "unittest", "discover", "-v"])
    if args.dry_run:
        print("PREFLIGHT PASS: no study execution requested (--dry-run).", flush=True)
        return

    if args.mode == "extended-paper":
        extended = [
            python,
            "extended_study.py",
            "--output-dir", str(output_dir),
            "--replications", str(args.replications),
            "--cv-folds", str(args.cv_folds),
            "--search-candidates", str(args.search_candidates),
            "--workers", str(args.workers),
        ]
        if args.resume:
            extended.append("--resume")
        if args.verbose:
            extended.append("--verbose")
        run_checked(extended)

    paper = [
        python,
        "paper_pipeline.py",
        "--output-dir", str(output_dir),
        "--replications", str(args.replications),
        "--bootstrap-resamples", str(args.bootstrap_resamples),
        "--jobs", str(args.workers),
        "--export-synthetic-data",
    ]
    if args.resume:
        paper.append("--resume")
    if args.verbose:
        paper.append("--verbose")
    run_checked(paper)
    run_checked([python, "verify_outputs.py", "--output-dir", str(output_dir)])
    if args.mode == "extended-paper":
        run_checked([python, "verify_extended_outputs.py", "--output-dir", str(output_dir)])
    print(f"COMPLETE: verified study artifacts are in {output_dir}", flush=True)


if __name__ == "__main__":
    main()

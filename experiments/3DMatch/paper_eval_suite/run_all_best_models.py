from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


SUITE_DIR = Path(__file__).resolve().parent
PAPER_TEST = SUITE_DIR / "paper_test.py"
PAPER_EVAL = SUITE_DIR / "paper_eval.py"
COMPARE = SUITE_DIR / "compare_variants.py"
AUDIT = SUITE_DIR / "audit_checkpoints.py"

ALL_VARIANTS = [
    "baseline", "spsa", "mspki", "pki", "full", "intended",
    "spsa_fine", "mspki_coarse", "pki_coarse", "swapped",
    "capacity_matched",
]


@dataclass(frozen=True)
class Condition:
    name: str
    benchmark: str
    rotation_mode: str = "none"
    keep_ratio: float = 1.0
    noise_std: float = 0.0


def _float_tag(value: float) -> str:
    return format(float(value), "g").replace(".", "p").replace("-", "m")


def _conditions_for_suite(suite: str) -> List[Condition]:
    standard = [
        Condition("standard_3DMatch", "3DMatch"),
        Condition("standard_3DLoMatch", "3DLoMatch"),
    ]
    rotated = [
        Condition("so3_3DMatch", "3DMatch", rotation_mode="so3"),
        Condition("so3_3DLoMatch", "3DLoMatch", rotation_mode="so3"),
    ]
    density = [
        Condition(
            f"density_{_float_tag(ratio)}_3DLoMatch",
            "3DLoMatch",
            keep_ratio=ratio,
        )
        for ratio in (0.75, 0.50, 0.25)
    ]
    noise = [
        Condition(
            f"noise_{_float_tag(sigma)}_3DLoMatch",
            "3DLoMatch",
            noise_std=sigma,
        )
        for sigma in (0.0025, 0.005, 0.01)
    ]

    if suite == "smoke":
        return [standard[0]]
    if suite == "standard":
        return standard
    if suite == "core":
        return standard + rotated
    if suite == "paper":
        return standard + rotated + density + noise
    if suite == "full":
        combined = [
            Condition(
                "density_0p5_noise_0p005_3DLoMatch",
                "3DLoMatch",
                keep_ratio=0.50,
                noise_std=0.005,
            ),
            Condition(
                "density_0p25_noise_0p01_3DLoMatch",
                "3DLoMatch",
                keep_ratio=0.25,
                noise_std=0.01,
            ),
        ]
        return standard + rotated + density + noise + combined
    raise ValueError(f"Unknown suite: {suite}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-command evaluation of baseline/SPSA/PKI/Full best checkpoints "
            "on standard, rotated, density, and noise conditions."
        )
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--metadata_root", required=True)
    parser.add_argument(
        "--checkpoint_root",
        required=True,
        help="Root containing baseline/full/pki/spsa run folders.",
    )
    parser.add_argument(
        "--results_root",
        required=True,
        help="Separate root for extracted features and paper evaluation results.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "spsa", "mspki", "full"],
        choices=ALL_VARIANTS,
    )
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--rotation_seed", type=int, default=7351)
    parser.add_argument("--perturbation_seed", type=int, default=7351)
    parser.add_argument("--re_feature_source", choices=["official", "decoder", "encoder"], default="official")
    parser.add_argument("--suite", choices=["smoke", "standard", "core", "paper", "full"], default="core")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["fhp", "ransac", "svd"],
        default=["fhp"],
    )
    parser.add_argument(
        "--corr_budgets",
        type=int,
        nargs="*",
        default=[50, 100, 250, 500],
        help="Applied only to RANSAC/SVD evaluations.",
    )
    parser.add_argument(
        "--estimator_scope",
        choices=["standard", "core", "all"],
        default="standard",
        help="Where non-FHP estimator/budget evaluations are run.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--knn_query_chunk_size", type=int, default=None)
    parser.add_argument("--knn_support_chunk_size", type=int, default=None)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--verbose_eval", action="store_true")
    parser.add_argument(
        "--save_full_features",
        action="store_true",
        help="Keep large point/feature arrays instead of compact paper-evaluation files.",
    )
    return parser


def _format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _run_command(
    command: Sequence[str],
    env: dict,
    log_file,
    dry_run: bool,
) -> int:
    rendered = _format_command(command)
    header = f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {rendered}\n"
    print(header, end="")
    log_file.write(header)
    log_file.flush()
    if dry_run:
        return 0

    process = subprocess.Popen(
        list(command),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_file.write(line)
    return_code = process.wait()
    log_file.write(f"[exit={return_code}]\n")
    log_file.flush()
    return return_code


def _checkpoint_path(args, variant: str) -> Path:
    return (
        Path(args.checkpoint_root)
        / variant
        / f"re_{args.re_feature_source}"
        / f"seed_{args.seed}"
        / "snapshots"
        / "best.pth.tar"
    )


def _condition_root(args, condition: Condition) -> Path:
    return Path(args.results_root) / condition.name


def _run_output_dir(args, condition: Condition, variant: str) -> Path:
    return (
        _condition_root(args, condition)
        / variant
        / f"re_{args.re_feature_source}"
        / f"seed_{args.seed}"
    )


def _benchmark_tag(condition: Condition, rotation_seed: int) -> str:
    if condition.rotation_mode == "none":
        return condition.benchmark
    return f"{condition.benchmark}_so3_seed{int(rotation_seed)}"


def _manifest_path(args, condition: Condition, variant: str) -> Path:
    return (
        _run_output_dir(args, condition, variant)
        / "features"
        / _benchmark_tag(condition, args.rotation_seed)
        / "manifest.json"
    )


def _summary_path(
    args,
    condition: Condition,
    variant: str,
    method: str,
    num_corr: Optional[int],
) -> Path:
    method_tag = method if num_corr is None else f"{method}_n{int(num_corr)}"
    return (
        _run_output_dir(args, condition, variant)
        / "registration"
        / _benchmark_tag(condition, args.rotation_seed)
        / f"summary_{method_tag}.json"
    )


def _manifest_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return bool(payload.get("completed", False))
    except (OSError, ValueError, TypeError):
        return False


def _base_config_args(args, condition: Condition, variant: str) -> List[str]:
    values = [
        "--variant",
        variant,
        "--seed",
        str(args.seed),
        "--dataset_root",
        str(Path(args.dataset_root)),
        "--metadata_root",
        str(Path(args.metadata_root)),
        "--output_root",
        str(_condition_root(args, condition)),
        "--num_workers",
        str(args.num_workers),
        "--rotation_mode",
        condition.rotation_mode,
        "--rotation_seed",
        str(args.rotation_seed),
        "--re_feature_source",
        args.re_feature_source,
    ]
    if args.knn_query_chunk_size is not None:
        values += ["--knn_query_chunk_size", str(args.knn_query_chunk_size)]
    if args.knn_support_chunk_size is not None:
        values += ["--knn_support_chunk_size", str(args.knn_support_chunk_size)]
    return values


def _estimator_condition_allowed(condition: Condition, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "core":
        return condition.name.startswith("standard_") or condition.name.startswith("so3_")
    return condition.name.startswith("standard_")


def _validate_paths(args) -> None:
    paths = {
        "project_root": Path(args.project_root),
        "dataset_root": Path(args.dataset_root),
        "metadata_root": Path(args.metadata_root),
        "checkpoint_root": Path(args.checkpoint_root),
    }
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
    for variant in args.variants:
        checkpoint = _checkpoint_path(args, variant)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {variant}: {checkpoint}")


def main() -> None:
    args = make_parser().parse_args()
    _validate_paths(args)
    conditions = _conditions_for_suite(args.suite)
    if args.suite == "smoke" and args.max_pairs is None:
        args.max_pairs = 10

    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    log_path = results_root / f"run_all_{time.strftime('%Y%m%d-%H%M%S')}.log"

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(args.project_root).resolve()),
            str((Path(args.project_root) / "experiments" / "3DMatch").resolve()),
            env.get("PYTHONPATH", ""),
        ]
    )

    failures = []
    with log_path.open("w", encoding="utf-8") as log_file:
        audit_command = [
            sys.executable,
            str(AUDIT),
            "--checkpoint_root",
            str(Path(args.checkpoint_root).resolve()),
            "--results_root",
            str(results_root),
            "--variants",
            *args.variants,
            "--seed",
            str(args.seed),
            "--re_feature_source",
            args.re_feature_source,
        ]
        audit_code = _run_command(audit_command, env, log_file, args.dry_run)
        if audit_code != 0:
            failures.append(("audit", "all", "checkpoints", audit_code))
            if not args.continue_on_error:
                raise SystemExit(audit_code)

        for condition in conditions:
            for variant in args.variants:
                checkpoint = _checkpoint_path(args, variant).resolve()
                manifest = _manifest_path(args, condition, variant)

                should_extract = args.overwrite or not _manifest_complete(manifest)
                if should_extract:
                    command = [
                        sys.executable,
                        str(PAPER_TEST),
                        *_base_config_args(args, condition, variant),
                        "--benchmark",
                        condition.benchmark,
                        "--snapshot",
                        str(checkpoint),
                        "--keep_ratio",
                        str(condition.keep_ratio),
                        "--noise_std",
                        str(condition.noise_std),
                        "--perturbation_seed",
                        str(args.perturbation_seed),
                        "--condition_name",
                        condition.name,
                    ]
                    if args.max_pairs is not None:
                        command += ["--max_pairs", str(args.max_pairs)]
                    if args.save_full_features:
                        command.append("--save_full_features")
                    if args.overwrite:
                        command.append("--overwrite_features")
                    return_code = _run_command(command, env, log_file, args.dry_run)
                    if return_code != 0:
                        failures.append((condition.name, variant, "extract", return_code))
                        if not args.continue_on_error:
                            raise SystemExit(return_code)
                        continue
                else:
                    message = f"[skip] completed extraction: {manifest}\n"
                    print(message, end="")
                    log_file.write(message)

                evaluations = []
                if "fhp" in args.methods:
                    evaluations.append(("fhp", None))
                if _estimator_condition_allowed(condition, args.estimator_scope):
                    for method in args.methods:
                        if method == "fhp":
                            continue
                        evaluations.append((method, None))
                        evaluations.extend((method, budget) for budget in args.corr_budgets)

                for method, num_corr in evaluations:
                    summary = _summary_path(args, condition, variant, method, num_corr)
                    if summary.is_file() and not args.overwrite:
                        message = f"[skip] existing summary: {summary}\n"
                        print(message, end="")
                        log_file.write(message)
                        continue
                    command = [
                        sys.executable,
                        str(PAPER_EVAL),
                        *_base_config_args(args, condition, variant),
                        "--benchmark",
                        condition.benchmark,
                        "--method",
                        method,
                    ]
                    if num_corr is not None:
                        command += ["--num_corr", str(num_corr)]
                    if args.max_pairs is not None:
                        command.append("--allow_partial")
                    if args.verbose_eval:
                        command.append("--verbose")
                    return_code = _run_command(command, env, log_file, args.dry_run)
                    if return_code != 0:
                        failures.append(
                            (condition.name, variant, f"eval:{method}:{num_corr}", return_code)
                        )
                        if not args.continue_on_error:
                            raise SystemExit(return_code)

        if "fhp" in args.methods:
            command = [
                sys.executable,
                str(COMPARE),
                "--results_root",
                str(results_root),
                "--seed",
                str(args.seed),
                "--re_feature_source",
                args.re_feature_source,
                "--method_tag",
                "fhp",
            ]
            return_code = _run_command(command, env, log_file, args.dry_run)
            if return_code != 0:
                failures.append(("aggregate", "all", "compare", return_code))

    print(f"\nRun log: {log_path}")
    print(f"Results root: {results_root}")
    if failures:
        print("Failures:")
        for failure in failures:
            print("  ", failure)
        raise SystemExit(1)
    print("All requested experiments completed successfully.")


if __name__ == "__main__":
    main()

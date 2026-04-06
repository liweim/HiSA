#!/usr/bin/env python3
"""Run HiSA on Spider2-V without the multi-method ``run_all.py`` entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import traceback
from typing import Dict, Iterable, List, Tuple


ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
HISA_ROOT = os.path.abspath(os.path.join(ADAPTER_DIR, "../.."))
SPIDER2V_ROOT = os.path.abspath(
    os.environ.get(
        "SPIDER2V_ROOT",
        ADAPTER_DIR
        if os.path.isdir(os.path.join(ADAPTER_DIR, "desktop_env", "envs"))
        else os.path.join(HISA_ROOT, "benchmarks", "Spider2-V"),
    )
)
for path in (HISA_ROOT, SPIDER2V_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from agents.hisa.main import default_pattern_namespace
from agents.run_hisa import run as run_hisa_tasks
from agents.utils import setup_logger, summary
from desktop_env.envs.desktop_env import DesktopEnv


logger = logging.getLogger("desktopenv")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end Spider2-V evaluation with HiSA"
    )

    # Environment configuration.
    parser.add_argument(
        "--path_to_vm",
        type=str,
        default="vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx",
    )
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--action_space", choices=["pyautogui"], default="pyautogui")
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--screen_width", type=int, default=1280)
    parser.add_argument("--screen_height", type=int, default=720)
    parser.add_argument("--client_password", type=str, default="password")

    # HiSA configuration.
    parser.add_argument("--global_planner_model", type=str, default="gpt-5")
    parser.add_argument("--visual_grounder_model", type=str, default="gta1-7b")
    parser.add_argument("--state_manager_model", type=str, default="gpt-5-mini")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--wo_pattern", action="store_true")
    parser.add_argument("--pattern_dir", type=str, default="./qdrant_storage")
    parser.add_argument("--pattern_namespace", type=str, default=None)
    parser.add_argument("--use_qdrant_server", action="store_true")
    parser.add_argument(
        "--qdrant_server_url", type=str, default="http://localhost:6333"
    )
    parser.add_argument("--wo_roi", action="store_true")
    parser.add_argument("--roi_margin", type=int, default=50)
    parser.add_argument("--refine_period", type=int, default=5)
    parser.add_argument("--bash_timeout", type=int, default=60)
    parser.add_argument("--wo_step", action="store_true")
    parser.add_argument("--wo_refinement", action="store_true")
    parser.add_argument("--sliding_window_size", type=int, default=5)

    # Evaluation and output configuration.
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default="evaluation_examples/test_abstract.json",
    )
    parser.add_argument(
        "--test_config_base_dir",
        type=str,
        default="evaluation_examples/examples",
    )
    parser.add_argument("--result_dir", type=str, default="./results/hisa")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun_fail", action="store_true")
    parser.add_argument("--get_score", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    args = parser.parse_args()
    if not args.pattern_namespace:
        args.pattern_namespace = default_pattern_namespace(args.result_dir)
    return args


def _all_tasks(test_meta: Dict[str, Iterable[str]]) -> List[Tuple[str, str]]:
    return [
        (domain, task_id)
        for domain, task_ids in test_meta.items()
        for task_id in task_ids
    ]


def _select_tasks(
    args: argparse.Namespace, test_meta: Dict[str, List[str]]
) -> List[Tuple[str, str]]:
    selected = []
    for domain, task_id in _all_tasks(test_meta):
        task_dir = os.path.join(args.result_dir, domain, task_id)
        result_path = os.path.join(task_dir, "result.txt")
        error_path = os.path.join(task_dir, "err_reason.txt")

        if args.rerun or not os.path.isfile(result_path) or os.path.isfile(error_path):
            selected.append((domain, task_id))
            continue
        if args.rerun_fail:
            try:
                with open(result_path, "r", encoding="utf-8") as file:
                    if float(file.read().strip()) <= 0.0:
                        selected.append((domain, task_id))
            except (OSError, ValueError):
                selected.append((domain, task_id))
    return selected


def _ensure_vm_resolution(
    env: DesktopEnv, width: int, height: int, active_logger: logging.Logger
) -> None:
    if width == 1920 and height == 1080:
        return

    script = textwrap.dedent(
        f"""
        import os
        import shutil
        import subprocess

        os.environ["DISPLAY"] = ":0"
        output = subprocess.check_output(
            "xrandr --query | awk '/ connected/{{print $1; exit}}'",
            shell=True,
            text=True,
        ).strip()
        if not output:
            raise RuntimeError("No connected display output found")

        mode = "{width}x{height}"
        modes = subprocess.check_output(
            "xrandr | awk '{{print $1}}'", shell=True, text=True
        ).split()
        if mode in modes:
            subprocess.check_call(["xrandr", "--output", output, "--mode", mode])
        else:
            if shutil.which("cvt") is None:
                raise RuntimeError(
                    f"Display mode {{mode}} is unavailable and cvt is not installed"
                )
            cvt_out = subprocess.check_output(
                "cvt {width} {height}", shell=True, text=True
            ).splitlines()
            modelines = [
                line for line in cvt_out if line.lstrip().startswith("Modeline ")
            ]
            if not modelines:
                raise RuntimeError(f"cvt did not produce a Modeline for {{mode}}")
            parts = modelines[0].split()
            name = parts[1].strip('"')
            subprocess.run(
                ["xrandr", "--newmode", name, *parts[2:]], check=False
            )
            subprocess.run(["xrandr", "--addmode", output, name], check=False)
            subprocess.check_call(["xrandr", "--output", output, "--mode", name])
        """
    ).strip()
    result = env.controller.run_python_script(script)
    if not result or result.get("status") == "error":
        error = (result or {}).get("error") or "guest returned no result"
        raise RuntimeError(f"Failed to set VM resolution: {error}")

    size = env.controller.get_vm_screen_size() or {}
    if size.get("width") != width or size.get("height") != height:
        raise RuntimeError(
            "VM resolution mismatch: "
            f"got {size.get('width')}x{size.get('height')}, "
            f"expected {width}x{height}"
        )
    active_logger.info("VM resolution set to %sx%s", width, height)


def _attach_resolution_guard(
    env: DesktopEnv, width: int, height: int, active_logger: logging.Logger
) -> None:
    original_reset = env.reset

    def guarded_reset(*args, **kwargs):
        observation = original_reset(*args, **kwargs)
        _ensure_vm_resolution(env, width, height, active_logger)
        if isinstance(observation, dict):
            observation["screenshot"] = env.controller.get_screenshot()
        return observation

    env.reset = guarded_reset


def main() -> None:
    global logger

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    result_name = os.path.basename(os.path.abspath(args.result_dir))
    logger = setup_logger(result_name, args.log_level)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as file:
        test_meta = json.load(file)
    if args.domain != "all":
        if args.domain not in test_meta:
            raise KeyError(
                f"Domain {args.domain!r} is not present in {args.test_all_meta_path}"
            )
        test_meta = {args.domain: test_meta[args.domain]}

    os.makedirs(args.result_dir, exist_ok=True)
    if args.get_score:
        summary(args.result_dir, test_meta)
        return

    tasks = _select_tasks(args, test_meta)
    logger.info("Tasks selected: %d", len(tasks))
    if not tasks:
        summary(args.result_dir, test_meta)
        return

    env = None
    try:
        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            snapshot_name=args.snapshot_name,
            headless=args.headless,
            require_a11y_tree=False,
        )
        _attach_resolution_guard(
            env, args.screen_width, args.screen_height, logger
        )
        args.env = env
        run_hisa_tasks(args, logger=logger, tasks=tasks)
    except Exception:
        logger.error("Spider2-V HiSA run failed:\n%s", traceback.format_exc())
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                logger.warning("Failed to close the Spider2-V environment: %s", exc)

    summary(args.result_dir, test_meta)


if __name__ == "__main__":
    main()

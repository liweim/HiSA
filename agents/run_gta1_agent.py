from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
from typing import List, Dict
from tqdm import tqdm
import traceback
from agents.gta1_agent import GTA1Agent
from agents.run_single import run_single_example
from agents.utils import setup_logger, save_args_to_settings, build_additional_contexts, summary

def config() -> argparse.Namespace:
    from desktop_env.desktop_env import DesktopEnv
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default="vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx")
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--record", action="store_true", help="Record the execution process")
    parser.add_argument("--action_space", type=str, default="pyautogui", help="Action type")
    parser.add_argument("--observation_type", type=str, default="screenshot", help="Observation type")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--screen_width", type=int, default=1920, help="Screen width")
    parser.add_argument("--screen_height", type=int, default=1080, help="Screen height")
    parser.add_argument("--client_password", type=str, default="password", help="Client password")

    # agent config
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples/examples"
    )

    # lm config
    parser.add_argument("--model", type=str, default="o3")
    parser.add_argument("--judge_model", type=str, default="o3")
    
    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_all.json"
    )

    # Flow control
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores only without running tests")

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results/gta1_agent",
                       help="Directory to save results")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help="Set the logging level")

    args = parser.parse_args()

    # Initialize environment
    args.env = DesktopEnv(
        path_to_vm=args.path_to_vm,
        snapshot_name=args.snapshot_name,
        headless=args.headless,
        require_a11y_tree=False
    )
    return args


def process_single_task(
    domain: str,
    task_id: str,
    cfg: dict,
    logger: logging.Logger,
    args: argparse.Namespace,
) -> tuple:
    """Process a single task."""
    logger.info(f"[Processing task] {domain}/{task_id}")

    # Setup result directory
    save_dir = os.path.join(args.result_dir, f"{domain}/{task_id}")

    # Construct example path
    example_path = os.path.join(args.test_config_base_dir, f"{domain}/{task_id}")

    # Build context
    additional_context = build_additional_contexts(
        example_dir=example_path,
        summarize_rag=False,
        use_rag=False,
    )

    try:
        # Initialize agent
        agent = GTA1Agent(
            max_steps=args.max_steps,
            client_password=args.client_password,
            planner_model=args.model,
            judge_model=args.judge_model,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
        )

        # Execute task
        logger.info(f"[Domain]: {domain}")
        logger.info(f"[Example ID]: {task_id}")
        logger.info(f"[Instruction]: {cfg['instruction'] + additional_context}")

        # Add domain to task config
        cfg['domain'] = domain

        # Create scores list
        scores = []

        # Run single example
        run_single_example(
            agent,
            args.env,
            cfg,
            args.max_steps,
            cfg['instruction'],
            additional_context,
            args,
            save_dir,
            scores,
        )

        score = scores[0] if scores else 0.0

        # Read execution log for statistics
        execution_log_path = os.path.join(save_dir, "execution_log.json")
        if os.path.exists(execution_log_path):
            with open(execution_log_path, "r", encoding="utf-8") as f:
                execution_log = json.load(f)
                stats = execution_log.get("statistics", {})
                total_cost = stats.get("total_cost", 0)

                logger.info(f"Task {domain}/{task_id} completed with score: {score}")
                logger.info(f"Total cost: ${total_cost:.4f}")
        else:
            logger.info(f"Task {domain}/{task_id} completed with score: {score}")

        return domain, score

    except Exception as e:
        logger.error(f"Error processing task {domain}/{task_id}")
        logger.error(traceback.format_exc())
        score = 0.0

        # Save error information
        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(str(score))
        with open(os.path.join(save_dir, "err_reason.txt"), "w") as f:
            f.write(f"Fatal error: {str(e)}")

        return domain, score


def run(args: argparse.Namespace, logger=None, tasks=None):
    """
    Run evaluation tasks.
    
    Args:
        args: Command line arguments
        logger: Logger instance (optional, will create if not provided)
        tasks: List of (domain, task_id) tuples (optional, will build from file if not provided)
    """
    # Setup logging configuration
    result_name = os.path.basename(args.result_dir)

    # Build tasks if not provided
    if tasks is None:
        with open(args.test_all_meta_path, encoding="utf-8") as f:
            test_all_meta = json.load(f)

        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}
        
        tasks = []
        for domain in test_all_meta:
            for task_id in test_all_meta[domain]:
                tasks.append((domain, task_id))

    if not args.get_score:
        if logger is None:
            logger = setup_logger(result_name, args.log_level)

        save_args_to_settings(args)

        tasks = []
        scores: Dict[str, List[float]] = {}

        # Process each domain and example
        for domain in test_all_meta:
            scores[domain] = []
            for task_id in test_all_meta[domain]:
                target_dir = os.path.join(args.result_dir, f"{domain}/{task_id}")
                result_path = os.path.join(target_dir, 'result.txt')
                cfg_path = os.path.join(args.test_config_base_dir, f"{domain}/{task_id}.json")
                if not os.path.exists(cfg_path):
                    cfg_path = os.path.join(args.test_config_base_dir, f"{domain}/{task_id}/{task_id}.json")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                # Clean up existing directory and prepare for execution (task filtering is done by run_all.py)
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                os.makedirs(target_dir, exist_ok=True)
                tasks.append((domain, task_id, cfg))

        # Execute all tasks
        if not tasks:
            logger.info("No tasks to process.")
        else:
            result_domain, score = process_single_task(domain, task_id, cfg, logger, args)
            results.append((result_domain, score))
            
            # Collect scores
            if result_domain not in scores:
                scores[result_domain] = []
            scores[result_domain].append(score)

if __name__ == "__main__":
    args = config()
    run(args)

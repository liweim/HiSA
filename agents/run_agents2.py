"""OSWorld's run.py with AgentS2."""

"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import traceback

from agents.gui_agents.s2.agents.agent_s import AgentS2
from agents.gui_agents.s2.agents.grounding import OSWorldACI
from tqdm import tqdm

from agents.run_single import run_single_example
from agents.utils import summary, save_args_to_settings, setup_logger, build_additional_contexts


def config() -> argparse.Namespace:
    from desktop_env.desktop_env import DesktopEnv

    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # lm config
    parser.add_argument("--model_provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL of the main generation model API.",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default="",
        help="The API key of the main generation model.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--stop_token", type=str, default=None)

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_all.json"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")

    # NEW!

    # Configuration 1
    parser.add_argument("--ground_provider", type=str, default="anthropic")
    parser.add_argument(
        "--ground_model", type=str, default="claude-3-7-sonnet-20250219"
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        default=1366,
        help="Width of screenshot image after processor rescaling",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        default=None,
        help="Height of screenshot image after processor rescaling",
    )

    # Configuration 2
    parser.add_argument("--endpoint_provider", type=str, default="")
    parser.add_argument("--endpoint_url", type=str, default="")
    parser.add_argument(
        "--endpoint_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )

    parser.add_argument("--kb_name", default="kb_s2", type=str)
    
    # RAG config
    parser.add_argument("--rag", action='store_true', help="Enable RAG context")
    parser.add_argument("--rag_topk", type=int, default=4)
    parser.add_argument("--summarize_rag", action='store_true', help="Summarize RAG context")
    parser.add_argument("--rag_filename", type=str, default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt")
    
    # Task control arguments
    parser.add_argument("--provider_name", type=str, default="vmware")
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument("--record", action="store_true", help="Record the execution")
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores without running tasks")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")

    args = parser.parse_args()

    # Create environment after checking get_score
    args.env = DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        snapshot_name=args.snapshot_name,
        headless=args.headless,
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
    )

    return args


def get_unfinished(target_dir, total_file_json):
    """Get unfinished tasks from the result directory"""
    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # Clean up incomplete results
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(result_dir, total_file_json):
    """Get current results from the result directory"""
    target_dir = result_dir
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


def run(args, logger=None, tasks=None):
    """
    Run evaluation tasks.
    
    Args:
        args: Command line arguments
        logger: Logger instance (optional, will create if not provided)
        tasks: List of (domain, task_id) tuples (optional, will build from file if not provided)
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Build tasks if not provided
    if tasks is None:
        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)
        
        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}
        
        tasks = []
        for domain in test_all_meta:
            for example_id in test_all_meta[domain]:
                tasks.append((domain, example_id))
    
    # If only getting scores, call summary and exit
    if args.get_score:
        summary(args.result_dir, tasks)
        sys.exit(0)
    
    # Setup logger
    if logger is None:
        logger = setup_logger(os.path.basename(args.result_dir), args.log_level)
    
    # Save settings
    save_args_to_settings(args)
    
    if not tasks:
        logger.info("No tasks to process.")
        sys.exit(0)
    
    # Initialize agent (shared across all tasks)
    engine_params = {
        "engine_type": args.model_provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
    }

    if args.endpoint_url:
        engine_params_for_grounding = {
            "engine_type": args.endpoint_provider,
            "base_url": args.endpoint_url,
            "api_key": args.endpoint_api_key,
        }
    else:
        grounding_height = (
            args.screen_height
            * args.grounding_width
            / args.screen_width
        )
        engine_params_for_grounding = {
            "engine_type": args.ground_provider,
            "model": args.ground_model,
            "grounding_width": args.grounding_width,
            "grounding_height": grounding_height,
        }

    grounding_agent = OSWorldACI(
        platform="linux",
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=args.screen_width,
        height=args.screen_height,
    )

    agent = AgentS2(
        engine_params,
        grounding_agent,
        platform="linux",
        action_space="pyautogui",
        observation_type="mixed",
        search_engine="LLM",
        memory_root_path=os.getcwd(),
        use_default_kb=True,
        memory_folder_name=args.kb_name,
        kb_release_tag="v0.2.2",
        embedding_engine_type="openai",
    )

    env = args.env
    max_steps = args.max_steps
    scores = []

    # Process each task
    for domain, example_id in tqdm(tasks, desc="Processing tasks"):
        config_file = os.path.join(
            args.test_config_base_dir, f"{domain}/{example_id}.json"
        )
        if not os.path.exists(config_file):
            config_file = os.path.join(
                args.test_config_base_dir, f"{domain}/{example_id}/{example_id}.json"
            )
        with open(config_file, "r", encoding="utf-8") as f:
            example = json.load(f)

        # Build context
        additional_context = build_additional_contexts(
            example_dir=os.path.dirname(config_file),
            summarize_rag=args.summarize_rag,
            use_rag=args.rag,
            rag_topk=args.rag_topk,
            rag_filename=args.rag_filename
        )

        logger.info(f"[Domain]: {domain}")
        logger.info(f"[Example ID]: {example_id}")
        instruction = example["instruction"] + additional_context 
        logger.info(f"[Instruction]: {instruction}")

        example_result_dir = os.path.join(
            args.result_dir,
            domain,
            example_id,
        )
        os.makedirs(example_result_dir, exist_ok=True)
        
        try:
            run_single_example(
                agent,
                env,
                example,
                max_steps,
                instruction,
                additional_context,
                args,
                example_result_dir,
                scores,
            )
        except Exception as e:
            logger.error(f"Exception in {domain}/{example_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Save error information
            with open(os.path.join(example_result_dir, "result.txt"), "w") as f:
                f.write("0.0")
            with open(os.path.join(example_result_dir, "err_reason.txt"), "w") as f:
                f.write(f"Fatal error: {str(e)}\n\n{traceback.format_exc()}")
            
            if args.record:
                env.controller.end_recording(
                    os.path.join(example_result_dir, "recording.mp4")
                )

    if env:
        env.close()
    
    if scores:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("No scores to report")
    
    # Summary accepts tasks list directly
    summary(args.result_dir, tasks)

if __name__ == "__main__":
    args = config()
    run(args)

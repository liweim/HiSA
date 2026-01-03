"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import json
import logging
import os
import shutil
import traceback
from tqdm import tqdm
from agents.run_single import run_single_example
from agents.gui_agents.s3.agents.agent_s import AgentS3
from agents.gui_agents.s3.agents.grounding import OSWorldACI
from utils import summary, save_args_to_settings, build_additional_contexts, setup_logger
from dotenv import load_dotenv

load_dotenv()

def config() -> argparse.Namespace:
    from desktop_env.desktop_env import DesktopEnv

    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument(
        "--provider_name",
        type=str,
        default="vmware",
        help="Virtualization provider (vmware, docker, aws, azure, gcp, virtualbox)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record the execution"
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
    parser.add_argument("--sleep_after_execution", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=8)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # lm config
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=1)

    # AgentS2 specific config
    parser.add_argument("--model_provider", type=str, default="openai")
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
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=1,
        help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)",
    )

    # grounding model config
    parser.add_argument(
        "--ground_provider",
        type=str,
        required=True,
        help="The provider for the grounding model",
    )
    parser.add_argument(
        "--ground_url", type=str, help="The URL of the grounding model"
    )
    parser.add_argument(
        "--ground_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )
    parser.add_argument(
        "--ground_model",
        type=str,
        required=True,
        help="The model name for the grounding model",
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        default=1920,
        help="Width of screenshot image after processor rescaling",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        default=1080,
        help="Height of screenshot image after processor rescaling",
    )

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_all.json"
    )
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores without running tasks")

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")
    
    # RAG config
    parser.add_argument("--rag", action='store_true', help="Enable RAG context")
    parser.add_argument("--rag_topk", type=int, default=4)
    parser.add_argument("--summarize_rag", action='store_true', help="Summarize RAG context")
    parser.add_argument("--rag_filename", type=str, default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt")
                       
    args = parser.parse_args()

    args.env = DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        snapshot_name=args.snapshot_name,
        headless=args.headless,
        require_a11y_tree=False,
        enable_proxy=False
    )

    return args


def get_unfinished(
    target_dir, total_file_json
):
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
                        # empty all files under example_id
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


def get_result(target_dir, total_file_json):
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
                        # empty all files under example_id
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
    ####### The complete version of the list of examples #######
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
    
    # If only getting scores, skip all execution
    if args.get_score:
        summary(args.result_dir, tasks)
        return
    
    save_args_to_settings(args)

    # Setup logger with result_name
    if logger is None:
        logger = setup_logger(os.path.basename(args.result_dir), args.log_level)

    if not tasks:
        logger.info("No tasks to process.")
        return
    
    # Build test_file_list from tasks
    test_file_list = {}
    for domain, example_id in tasks:
        if domain not in test_file_list:
            test_file_list[domain] = []
        test_file_list[domain].append(example_id)
    
    # Initialize agent (shared across all tasks)
    engine_params = {
        "engine_type": args.model_provider,
        "model": args.model,
        "base_url": getattr(args, "model_url", ""),
        "api_key": getattr(args, "model_api_key", ""),
        "temperature": getattr(args, "model_temperature", None),
    }
    engine_params_for_grounding = {
        "engine_type": args.ground_provider,
        "model": args.ground_model,
        "base_url": getattr(args, "ground_url", ""),
        "api_key": getattr(args, "ground_api_key", ""),
        "grounding_width": args.grounding_width,
        "grounding_height": args.grounding_height,
    }

    grounding_agent = OSWorldACI(
        env=args.env,
        platform="linux",
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=args.screen_width,
        height=args.screen_height,
    )
    agent = AgentS3(
        engine_params,
        grounding_agent,
        platform="linux",
        max_trajectory_length=args.max_trajectory_length,
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
        
        # Reset grounding agent usage stats before each task
        grounding_agent.grounding_model.engine.reset_stats()
        
        try:
            run_single_example(
                agent,
                env,
                example,
                max_steps,
                example["instruction"],
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
import argparse
import os
import sys
import json
from typing import List
from tqdm import tqdm
import logging
import textwrap
import subprocess

OSWORLD_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OSWORLD_ROOT, "../.."))

for path in (PROJECT_ROOT, OSWORLD_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from desktop_env.desktop_env import DesktopEnv

from utils import summary, setup_logger

# Global variables
logger = None  # Will be initialized in run()

def filter_tasks(args, test_all_meta: dict, logger) -> List[tuple]:
    """
    Filter tasks based on rerun/rerun_fail flags.
    
    Returns:
        List of (domain, example_id) tuples to execute
    """
    tasks_to_run = []
    
    for domain in test_all_meta:
        for example_id in test_all_meta[domain]:
            target_dir = os.path.join(args.result_dir, domain, example_id)
            execution_log_path = os.path.join(target_dir, 'execution_log.json')
            result_path = os.path.join(target_dir, 'result.txt')
            err_reason_path = os.path.join(target_dir, 'err_reason.txt')
            if not os.path.exists(execution_log_path):
                if os.path.exists(result_path):
                    os.remove(result_path)
            
            should_skip = False
            if not args.rerun and os.path.exists(result_path) and not os.path.exists(err_reason_path):
                try:
                    result = float(open(result_path, 'r').read().strip())
                    # Skip successful tasks, or failed tasks if not rerun_fail
                    if result > 0.0 or not args.rerun_fail:
                        should_skip = True
                except (ValueError, IOError) as e:
                    logger.warning(f"Failed to read result for {domain}/{example_id}: {e}")
            
            if not should_skip:
                tasks_to_run.append((domain, example_id))
    
    return tasks_to_run


def _ensure_vm_resolution(env, width: int, height: int, logger: logging.Logger) -> None:
    if width == 1920 and height == 1080:
        return
        
    script = textwrap.dedent(f"""
        import os
        import subprocess

        os.environ["DISPLAY"] = ":0"
        output = subprocess.check_output(
            "xrandr --query | awk '/ connected/{{print $1; exit}}'",
            shell=True,
            text=True
        ).strip()
        if not output:
            raise RuntimeError("No connected display output found")

        mode = "{width}x{height}"
        modes = subprocess.check_output("xrandr | awk '{{print $1}}'", shell=True, text=True).split()
        if mode in modes:
            subprocess.check_call(["xrandr", "--output", output, "--mode", mode])
        else:
            if subprocess.call("command -v cvt >/dev/null 2>&1", shell=True) != 0:
                raise RuntimeError("cvt not found; install x11-xserver-utils in the VM")
            cvt_out = subprocess.check_output(
                "cvt {width} {height}",
                shell=True,
                text=True
            ).splitlines()
            if len(cvt_out) < 2:
                raise RuntimeError("cvt output is invalid")
            parts = cvt_out[1].split()
            if len(parts) < 3 or parts[0] != "Modeline":
                raise RuntimeError("Unexpected cvt output: " + cvt_out[1])
            name = parts[1].strip('"')
            params = parts[2:]
            subprocess.call(["xrandr", "--newmode", name, *params])
            subprocess.call(["xrandr", "--addmode", output, name])
            subprocess.check_call(["xrandr", "--output", output, "--mode", name])
    """).strip()

    try:
        result = env.controller.run_python_script(script)
    except Exception as exc:
        raise SystemExit(f"Failed to set VM resolution: {exc}")

    if result and result.get("status") == "error":
        raise SystemExit(f"Failed to set VM resolution: {result.get('error')}")

    size = env.controller.get_vm_screen_size() or {}
    if size.get("width") != width or size.get("height") != height:
        raise SystemExit(
            f"VM resolution mismatch: got {size.get('width')}x{size.get('height')}, expected {width}x{height}"
        )
    logger.info(f"VM resolution set to {width}x{height}")


def _attach_resolution_guard(env, width: int, height: int, logger: logging.Logger) -> None:
    """Ensure VM resolution after every env.reset call."""
    original_reset = env.reset

    def guarded_reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        _ensure_vm_resolution(env, width, height, logger)
        return result

    env.reset = guarded_reset


def run():
    parser = argparse.ArgumentParser(description="Run evaluation for agent framework")

    # ==================== Common Arguments (used by all methods) ====================
    parser.add_argument(
        "--method", type=str, default="hisa", help="Method to use"
    )

    # Environment config
    parser.add_argument(
        "--provider_name", type=str, default="vmware", help="Provider name"
    )
    parser.add_argument(
        "--path_to_vm",
        type=str,
        default="./vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
        help="Path to VM file",
    )
    parser.add_argument("--snapshot_name", type=str, default="low_res")
    parser.add_argument("--screen_width", type=int, default=1280)
    parser.add_argument("--screen_height", type=int, default=720)
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument(
        "--client_password", type=str, default="password", help="VM client password"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode or machine"
    )
    parser.add_argument("--record", action="store_true", help="Record the execution process")
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )

    # Task/example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default=os.path.join("evaluation_examples", "test_one.json"),
    )
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples/examples"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="Rerun tests that have already been run"
    )
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores")

    # Output/logging config
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results/dual_agent",
        help="Directory to save results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    # Verbose instruction config
    parser.add_argument(
        "--verbose_instruction",
        action="store_true",
        help="Enable verbose instruction loading",
    )

    # Shared arguments (used by multiple methods)
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model (spider2v_agent, agents2, agents3, gta1)")
    parser.add_argument("--max_steps", type=int, default=15, help="Maximum steps (agents2, agents3, hisa, gta1)")
    parser.add_argument("--max_trajectory_length", type=int, default=8, help="Max trajectory length (spider2v_agent default: 3, agents3)")
    parser.add_argument("--model_provider", type=str, default="openai", help="Model provider (agents2, agents3)")
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL of the main generation model API (agents2, agents3)",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default="",
        help="The API key of the main generation model (agents2, agents3)",
    )
    parser.add_argument(
        "--ground_provider",
        type=str,
        help="The provider for the grounding model (agents2, agents3)",
    )
    parser.add_argument("--ground_url", type=str, help="The URL of the grounding model (agents2, agents3)")
    parser.add_argument(
        "--ground_api_key",
        type=str,
        default="",
        help="The API key of the grounding model (agents2, agents3)",
    )
    parser.add_argument(
        "--ground_model",
        type=str,
        help="The model name for the grounding model (agents2, agents3)",
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        default=1280,
        help="Width of screenshot image after processor rescaling (agents2, agents3)",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        default=720,
        help="Height of screenshot image after processor rescaling (agents2, agents3)",
    )

    # ==================== Spider2V Agent (agents.run_spider2v_agent) Arguments ====================
    parser.add_argument("--temperature", type=float, default=1, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top p")
    parser.add_argument("--max_tokens", type=int, default=1500, help="Max tokens")
    parser.add_argument("--stop_token", type=str, default=None, help="Stop token")
    parser.add_argument(
        "--observation_space",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="som",
        help="Observation space",
    )
    parser.add_argument("--execution_feedback", action="store_true", help="Use execution feedback")
    parser.add_argument("--a11y_tree_max_tokens", type=int, default=5000, help="A11y tree max tokens")
    parser.add_argument("--example", type=str, default=os.path.join("evaluation_examples", "test_one.json"), help="Example file")
    parser.add_argument("--rag", action="store_true", help="Enable RAG context")
    parser.add_argument("--rag_topk", type=int, default=4, help="Top k to use for RAG")
    parser.add_argument(
        "--summarize_rag", action="store_true", help="Summarize RAG context"
    )
    parser.add_argument(
        "--rag_filename",
        type=str,
        default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt",
        help="RAG retrieved context file name",
    )

    # ==================== Coact (agents.run_coact) Arguments ====================
    parser.add_argument(
        "--oai_config_path",
        type=str,
        default="/your/path/to/OAI_CONFIG_LIST",
        help="OAI config path",
    )
    parser.add_argument("--orchestrator_model", type=str, default="o3-2025-04-16", help="Orchestrator model")
    parser.add_argument("--coding_model", type=str, default="o4-mini-2025-04-16", help="Coding model")
    parser.add_argument("--summarizer_model", type=str, default="o4-mini-2025-04-16", help="Summarizer model")
    parser.add_argument("--cua_model", type=str, default="computer-use-preview", help="CUA model")
    parser.add_argument("--orchestrator_max_steps", type=int, default=15, help="Orchestrator max steps")
    parser.add_argument("--coding_max_steps", type=int, default=20, help="Coding max steps")
    parser.add_argument("--cua_max_steps", type=int, default=25, help="CUA max steps")
    parser.add_argument("--cut_off_steps", type=int, default=50, help="Cut off steps")

    # ==================== Agents2 (agents.run_agents2) Arguments ====================
    parser.add_argument("--endpoint_provider", type=str, default="", help="Endpoint provider")
    parser.add_argument("--endpoint_url", type=str, default="", help="Endpoint URL")
    parser.add_argument(
        "--endpoint_api_key",
        type=str,
        default="",
        help="The API key of the grounding model",
    )
    parser.add_argument("--kb_name", default="kb_s2", type=str, help="Knowledge base name for Agent S2")

    # ==================== Agents3 (agents.run_agents3) Arguments ====================
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=1,
        help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)",
    )

    # ==================== HiSA (agents.run_hisa) Arguments ====================
    parser.add_argument(
        "--global_planner_model",
        type=str,
        default="qwen3.5-9b",
        help="Model for Global Planner agent",
    )
    parser.add_argument(
        "--visual_grounder_model",
        type=str,
        default="gta1-7b",
        help="Model for Visual Grounder agent",
    )
    parser.add_argument(
        "--visual_grounder_scale",
        type=float,
        default=1.0,
        help="Scale factor for visual grounder image preprocessing",
    )
    parser.add_argument("--state_manager_model", type=str, default="qwen3.5-9b",
                       help="Model for auxiliary tasks (step abstraction, context refinement, pattern induction, etc.)")
    parser.add_argument("--wo_pattern", action="store_true", help="Disable pattern induction (pattern induction is enabled by default)")
    parser.add_argument("--wo_roi", action="store_true",
                       help="Disable ROI cropping (ROI cropping is enabled by default, reduces token usage)")
    parser.add_argument("--roi_margin", type=int, default=50,
                       help="Margin around ROI when cropping (default: 50)")
    parser.add_argument("--refine_period", type=int, default=5,
                       help="Period to refine (default: 5)")
    parser.add_argument("--bash_timeout", type=int, default=60,
                       help="Timeout for bash script execution in seconds (default: 300)")
    parser.add_argument("--wo_step", action="store_true",
                       help="Skip step abstraction and use full conversation history")
    parser.add_argument("--wo_refinement", action="store_true",
                       help="Disable context refinement and use sliding window")
    parser.add_argument("--sliding_window_size", type=int, default=5,
                       help="Sliding window size (number of conversation turns to keep) (default: 5)")
    parser.add_argument("--pattern_dir", type=str, default="./qdrant_storage", help="Qdrant storage directory")
    parser.add_argument("--use_qdrant_server", action="store_true", help="Use Qdrant server, otherwise use local file storage")
    parser.add_argument("--qdrant_server_url", type=str, default="http://localhost:6333", help="Qdrant server URL")

    # ==================== GTA1 Agent (agents.run_gta1_agent) Arguments ====================
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Judge model")

    args = parser.parse_args()

    # Setup logger using utils.setup_logger
    global logger
    result_name = os.path.basename(args.result_dir)
    logger = setup_logger(result_name, args.log_level)

    # Single environment mode - use unified task filtering
    # Load test metadata
    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    
    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}
    
    # If only getting scores, skip execution
    if args.get_score:
        
        summary(args.result_dir, test_all_meta)
        sys.exit(0)
    
    # Filter tasks using unified logic
    tasks_to_run = filter_tasks(args, test_all_meta, logger)
    
    if not tasks_to_run:
        logger.info("No tasks to process. All tasks have already been completed.")
        summary(args.result_dir, test_all_meta)
        sys.exit(0)
    
    # Log tasks info
    task_count_by_domain = {}
    for domain, example_id in tasks_to_run:
        if domain not in task_count_by_domain:
            task_count_by_domain[domain] = 0
        task_count_by_domain[domain] += 1
    
    left_info = ""
    for domain, count in task_count_by_domain.items():
        left_info += f"{domain}: {count}\n"
    logger.info(f"Tasks to process:\n{left_info}")
    
    # Create environment
    env = DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        snapshot_name=args.snapshot_name,
        headless=args.headless,
        require_a11y_tree=False,
        enable_proxy=False,
        screen_size=(args.screen_width, args.screen_height),
    )
    args.env = env

    _attach_resolution_guard(
        args.env,
        args.screen_width,
        args.screen_height,
        logger,
    )

    # Import run function
    if args.method == "coact":
        from agents.run_coact import run
    elif args.method == "agents3":
        from agents.run_agents3 import run
    elif args.method == "hisa":
        from agents.run_hisa import run
    elif args.method == "hisa1":
        from agents.run_hisa1 import run
    elif args.method == "hisa2":
        from agents.run_hisa2 import run
    elif args.method == "hisa3":
        from agents.run_hisa3 import run
    elif args.method == "gta1":
        from agents.run_gta1_agent import run
    else:
        raise ValueError(f"Invalid method: {args.method}")
    
    # Execute tasks one by one
    for domain, example_id in tqdm(tasks_to_run, desc="Processing tasks"):
        logger.info(f"Processing {domain}/{example_id} for method {args.result_dir}")
        try:
            # Create tasks list with only this task
            single_task = [(domain, example_id)]
            run(args, logger=logger, tasks=single_task)
        except Exception as e:
            logger.error(f"Error processing {domain}/{example_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Save error information
            example_result_dir = os.path.join(args.result_dir, domain, example_id)
            os.makedirs(example_result_dir, exist_ok=True)
            with open(os.path.join(example_result_dir, "result.txt"), "w") as f:
                f.write("0.0")
            with open(os.path.join(example_result_dir, "err_reason.txt"), "w") as f:
                f.write(f"Fatal error: {str(e)}\n\n{traceback.format_exc()}")
    
    # Cleanup
    try:
        try:
            env.close()
        except Exception as close_error:
            error_msg = str(close_error)
            if "not powered on" in error_msg or "not running" in error_msg:
                logger.info("VM already stopped, skipping close")
            else:
                logger.warning(f"Error closing environment: {close_error}")
        
        # Clean VMware lock files
        vm_dir = os.path.dirname(args.path_to_vm)
        try:
            env.clean_lock(vm_dir)
            logger.info("Lock files cleaned")
        except Exception as lock_error:
            logger.debug(f"Error cleaning locks: {lock_error}")
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}")
    
    # Show summary
    summary(args.result_dir, test_all_meta)

if __name__ == "__main__":
    run()

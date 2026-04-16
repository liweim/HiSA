"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse
import json
import logging
import os
import shutil
import traceback
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
from agents.spider2v_agent import PromptAgent
from agents.utils import summary, save_args_to_settings, setup_logger


def get_retrieved_context(config_path: str, topk: int = 4, file_name: str = "retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt") -> str:
    context_path = os.path.join(os.path.dirname(config_path), file_name)
    if os.path.exists(context_path):
        with open(context_path, "r", encoding="utf-8") as f:
            context = f.read().strip()
        if context.strip() == "": return None
        splits = context.split("Documentation Source:")
        if len(splits) > topk + 1: # the first is ""
            return "Documentation Source:".join(splits[:topk + 1])
        return context
    raise ValueError(f"Retrieved context not found under {os.path.dirname(config_path)}")

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on the benchmark")

    # environment config
    parser.add_argument('-p', "--path_to_vm", type=str, help="path to the VM executable .vmx file, if None, automatically find the VM in vm_data/ folder")
    parser.add_argument('-s', "--snapshot_name", type=str, default="init_state", help="Snapshot name to use (overwrite snapshot in each example config)")
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument(
        "--action_space",
        choices=[
            "pyautogui",
            "computer_13"
        ],
        default="pyautogui",
        help="Action space to use for the agent"
    )
    parser.add_argument(
        "--observation_space",
        choices=[
            "screenshot",
            "a11y_tree",
            "screenshot_a11y_tree",
            "som"
        ],
        default="som",
        help="Observation space to use for the environment",
    )
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=15, help="Maximum number of steps for each example, this can be altered dynamically according to field `action_number` in the example config")

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3, help='maximum length of interaction history to provide to the agent')
    parser.add_argument("--a11y_tree_max_tokens", type=int, default=5000, help='maximum length of interaction history to provide to the agent')

    # llm config
    parser.add_argument('-m', "--model", type=str, default="gpt-4o-2024-05-13", help="LLM model to use for the agent")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)

    # example config
    parser.add_argument('-e', "--example", type=str, default=os.path.join('evaluation_examples', 'test_one.json'), help="JSON dict containing example ids to run")
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples/examples"
    )
    parser.add_argument("--exclude_account", action='store_true')
    parser.add_argument("--execution_feedback", action='store_true', help="whether to use execution feedback for the agent")
    parser.add_argument("--rag", action='store_true', help="Whether to use RAG for the agent")
    parser.add_argument("--rag_topk", type=int, default=4, help="Top k to use for RAG")
    parser.add_argument("--rag_filename", type=str, default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt", help="RAG retrieved context file name")
    parser.add_argument("--verbose_instruction", action='store_true', help="Enable verbose instruction loading")
    parser.add_argument("--domains", choices=['all'], nargs='+', default=["all"], help="Application names list to filter examples")

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results/som_gpt_4o_rag_ef")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help="Set the logging level")
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores")
    args = parser.parse_args()

    if args.observation_space == 'som':
        assert args.action_space == 'pyautogui', "SOM only supports pyautogui action space"
    return args



def process_single_task(
    domain: str,
    task_id: str,
    example_dict: dict,
    logger: logging.Logger,
    env,
    agent: PromptAgent,
    args: argparse.Namespace,
) -> Tuple[str, float]:
    """Process a single task with the agent."""
    config_file = example_dict['config']
    result_dir = example_dict['result']

    logger.info(f"[Processing task] {domain}/{task_id}")

    try:
        # Load example config
        with open(config_file, "r", encoding="utf-8") as f:
            example = json.load(f)

        # Build context
        if args.rag:
            example['context'] = get_retrieved_context(config_file, args.rag_topk, file_name=args.rag_filename)
        else:
            example['context'] = None

        logger.info(f"[Domain]: {domain}")
        logger.info(f"[Example ID]: {task_id}")
        logger.info(f"[Instruction]: {example['instruction']}")

        # Execute task
        score = run_single.run_single_example(agent, env, example, result_dir, args)

        # Save results
        with open(os.path.join(result_dir, "result.txt"), "w") as f:
            f.write(str(score))

        # Read trajectory log for statistics
        trajectory_log_path = os.path.join(result_dir, "trajectory.jsonl")
        if os.path.exists(trajectory_log_path):
            total_tokens = 0
            total_cost = 0.0
            num_steps = 0

            with open(trajectory_log_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            num_steps += 1
                            # Extract token usage if available
                            if 'token_usage' in entry:
                                usage = entry['token_usage']
                                if isinstance(usage, dict):
                                    total_tokens += usage.get('total_tokens', 0)
                            # Extract cost if available
                            if 'cost' in entry:
                                total_cost += entry['cost']
                        except json.JSONDecodeError:
                            continue

            logger.info(f"Task {domain}/{task_id} completed with score: {score}")
            logger.info(f"Total steps: {num_steps}")
            if total_tokens > 0:
                logger.info(f"Total tokens: {total_tokens}")
            if total_cost > 0:
                logger.info(f"Total cost: ${total_cost:.4f}")
        else:
            logger.info(f"Task {domain}/{task_id} completed with score: {score}")

        return domain, score

    except Exception as e:
        logger.error(f"Error processing task {domain}/{task_id}")
        logger.error(traceback.format_exc())
        score = 0.0

        # Save error information
        with open(os.path.join(result_dir, "result.txt"), "w") as f:
            f.write(str(score))
        with open(os.path.join(result_dir, "err_reason.txt"), "w") as f:
            f.write(f"Fatal error: {str(e)}\n\n{traceback.format_exc()}")

        return domain, 0.0


def run(args: argparse.Namespace, logger=None, tasks=None):
    """
    Main execution function.
    
    Args:
        args: Command line arguments
        logger: Logger instance (optional, will create if not provided)
        tasks: List of (domain, task_id) tuples (optional, will build from file if not provided)
    """
    result_name = os.path.basename(args.result_dir)

    # Build tasks if not provided
    if tasks is None:
        with open(args.example, encoding="utf-8") as f:
            test_all_meta = json.load(f)
        
        tasks = []
        for domain in test_all_meta:
            for task_id in test_all_meta[domain]:
                tasks.append((domain, task_id))

    if not args.get_score:
        if logger is None:
            logger = setup_logger(result_name, args.log_level)
        save_args_to_settings(args, args.result_dir)

        # Build examples from tasks
        examples = []
        for domain, task_id in tasks:
            target_dir = os.path.join(args.result_dir, f"{domain}/{task_id}")
            cfg = os.path.join(args.test_config_base_dir, f"{domain}/{task_id}/{task_id}.json")
            example = {
                "id": task_id,
                "domain": domain,
                "config": cfg,
                "result": target_dir,
                "action_number": json.load(open(cfg, 'r'))["action_number"]
            }
            examples.append(example)

        if not examples:
            logger.info("No tasks to process.")
        else:
            env = args.env
            # Initialize agent
            agent = PromptAgent(
                platform="ubuntu",
                model=args.model,
                max_tokens=args.max_tokens,
                action_space=args.action_space,
                observation_space=args.observation_space,
                execution_feedback=args.execution_feedback,
                screen_size=(args.screen_width, args.screen_height),
                temperature=args.temperature,
                max_trajectory_length=args.max_trajectory_length,
                a11y_tree_max_tokens=args.a11y_tree_max_tokens
            )

            try:
                scores: Dict[str, List[float]] = {}
                results = []

                for example in tqdm(examples, desc="Processing tasks"):
                    domain = example['domain']
                    task_id = example['id']

                    if domain not in scores:
                        scores[domain] = []

                    result = process_single_task(domain, task_id, example, logger, env, agent, args)
                    results.append(result)

                # Collect scores from results
                for domain, score in results:
                    scores[domain].append(score)

            finally:
                # Always cleanup environment
                try:
                    env.close()
                except Exception as cleanup_error:
                    logger.warning(f"Error during cleanup: {cleanup_error}")

    # Calculate and display final results
    # Summary accepts tasks list directly
    summary(args.result_dir, tasks)

def get_examples(args, test_all_meta, logger: logging.Logger = None, easy_first: bool = True) -> List[Dict[str, str]]:
    """ Get [Filter] the list of example dict for the current experiment.
    # Filter method:
    - args.rerun (bool): if True, rerun tests that have already been run
    - args.rerun_fail (bool): if True, rerun failed tests
    - args.domains (List[str]): if not contain "all", only include examples under the specified domains
    - args.exclude_account (bool): if True, exclude examples that are related to real accounts
    - easy_first (bool): if True, sort examples that are easy to run first (smaller action_number)

    # The returned dict for each example in the List containing:
        - id: example id
        - domain: example domain, a.k.a., professional tool name
        - config: .json config path for the example
        - result: path to the result directory for the example
            note that, the result directory will also be reset implicitly
    """
    # Create a default logger if none provided
    if logger is None:
        logger = logging.getLogger(__name__)

    examples_to_run = []
    for domain in test_all_meta:
        for ex_id in test_all_meta[domain]:
            target_dir = os.path.join(args.result_dir, f"{domain}/{ex_id}")
            result_path = os.path.join(target_dir, 'result.txt')
            cfg = os.path.join(args.test_config_base_dir, f"{domain}/{ex_id}/{ex_id}.json")

            # Check if we should skip this task
            should_skip = False
            if not args.rerun and os.path.exists(result_path) and not os.path.exists(os.path.join(target_dir, 'err_reason.txt')):
                result = float(open(result_path, 'r').read())
                logger.info(f"Results already exist in {domain}/{ex_id}, result: {result}")

                # Skip successful tasks, or skip failed tasks if not rerun_fail
                if result > 0.0 or not args.rerun_fail:
                    should_skip = True

            if not should_skip:
                # Clean up existing directory and add to tasks
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                os.makedirs(target_dir, exist_ok=True)
                if args.observation_space != "a11y_tree":
                    os.makedirs(os.path.join(target_dir, "screenshots"), exist_ok=True)
                if args.observation_space != "screenshot":
                    os.makedirs(os.path.join(target_dir, "a11y_trees"), exist_ok=True)
                example = {
                    "id": ex_id,
                    "domain": domain,
                    "config": cfg,
                    "result": target_dir,
                    "action_number":  json.load(open(cfg, 'r'))["action_number"]
                }
                examples_to_run.append(example)

    logger.info(f"Total examples to run: {len(examples_to_run)}")
    if easy_first:
        examples_to_run = sorted(examples_to_run, key=lambda x: x['action_number'])
    return examples_to_run


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    run(args)
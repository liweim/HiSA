import argparse
import base64
import glob
import shutil
import traceback
from typing import Dict, List
import json
import time
import os
from agents.coact.operator_agent import OrchestratorAgent, OrchestratorUserProxyAgent
from agents.coact.autogen import LLMConfig
from multiprocessing import cpu_count
from agents.utils import build_additional_contexts, serialize_json, save_args_to_settings, get_price, setup_logger, count_images_in_messages
from tqdm import tqdm
import nest_asyncio
nest_asyncio.apply()

def filter_base64_images_from_messages(messages: List[dict]) -> List[dict]:
    """Remove base64 images from messages for logging purposes."""
    filtered_messages = []
    for msg in messages:
        filtered_msg = {"role": msg["role"]}
        content = msg["content"]
        
        if isinstance(content, str):
            # String content - keep as is
            filtered_msg["content"] = content
        elif isinstance(content, list):
            # List content - filter out images
            filtered_content = []
            for item in content:
                if item.get("type") == "input_text":
                    filtered_content.append(item)
                elif item.get("type") == "input_image":
                    # Replace image with placeholder
                    filtered_content.append({
                        "type": "input_image",
                    })
            filtered_msg["content"] = filtered_content
        else:
            filtered_msg["content"] = content
        
        filtered_messages.append(filtered_msg)
    
    return filtered_messages


def merge_conversation_and_actions(conversation_history: List[dict], action_logs: List[dict]) -> List[dict]:
    """Merge conversation messages and action logs into a unified timeline."""
    timeline = []
    
    # Filter conversation messages to remove base64 images
    filtered_messages = filter_base64_images_from_messages(conversation_history)
    
    # Group messages into pairs (user + assistant)
    message_pairs = []
    i = 0
    while i < len(filtered_messages):
        if filtered_messages[i]["role"] == "user":
            user_msg = filtered_messages[i]
            assistant_msg = filtered_messages[i + 1] if i + 1 < len(filtered_messages) else None
            message_pairs.append({"user": user_msg, "assistant": assistant_msg})
            i += 2
        else:
            i += 1
    
    # Match conversation pairs with action logs
    # First pair is initial task instruction (step 0)
    if message_pairs:
        first_pair = message_pairs[0]
        timeline.append({
            "step": 0,
            "type": "task_instruction",
            "user_message": first_pair["user"]["content"],
            "assistant_response": first_pair["assistant"]["content"] if first_pair["assistant"] else None
        })
    
    # Process remaining pairs with corresponding actions
    for idx, pair in enumerate(message_pairs[1:], start=1):
        # Add action log if available
        if idx - 1 < len(action_logs):
            action = action_logs[idx - 1].copy()
            # Add conversation context to action
            action["coordinator_decision"] = pair["assistant"]["content"] if pair["assistant"] else None
            action["execution_result"] = pair["user"]["content"] if idx < len(message_pairs) else None
            timeline.append(action)
        else:
            # No action log, just record the conversation
            timeline.append({
                "step": idx,
                "type": "conversation_only",
                "user_message": pair["user"]["content"],
                "assistant_response": pair["assistant"]["content"] if pair["assistant"] else None
            })
    
    # Add any remaining action logs that don't have conversation pairs
    for idx in range(len(message_pairs) - 1, len(action_logs)):
        if idx < len(action_logs):
            timeline.append(action_logs[idx])
    
    return timeline


TASK_DESCRIPTION = """# Your role
You are a task solver, you need to complete a computer-using task step-by-step.
1. Describe the screenshot.
2. Provide a detailed plan, including a list of user requirements like specific file name, file path, etc.
3. Follow the following instructions and complete the task with your skills.
    - If you think the task is impossible to complete (no file, wrong environment, etc.), reply with "INFEASIBLE" to end the conversation.
    - **Do not** do (or let coding/GUI agent do) anything else out of the user's instruction like change the file name. This will make the task fail.
    - Check every screenshot carefully and see if it fulfills the task requirement.
    - You MUST try the Coding Agent first for file operation tasks like spreadsheet modification.
4. Verify the result and see if it fulfills the user's requirement.

# Your helpers
You can use the following tools to solve the task. You can only call one of gui agent or coding agent per reply:

## Programmer
Let a programmer to solve a subtask you assigned. 
The Programmer can write python or bash code to modify almost everything in the computer, like files, apps, system settings, etc. 
It requires a environment description and a detailed task description. As detailed as possible.
Can use any python package you instructed.
Will return a summary with the output of the code.
When letting coding agent to modify the spreadsheet, after the task completed, you MUST make sure EVERY modified value in the spreadsheet is in the desired position (e.g., filled in the expected cell) by a GUI Operator.
After that, if anything is wrong, tell the programmer to modify it.

## GUI Operator
Let a GUI agent to solve a subtask you assigned. 
GUI agent can operate the computer by clicking and typing (but not accurate). 
Require a detailed task description.
When you call GUI agent, it will only have a **20-step** budget to complete your task. Each step is a one-time interaction with OS like mouse click or keyboard typing. Please take this into account when you plan the actions.
If you let GUI Operator to check the result, you MUST let it close and reopen the file because programmer's result will NOT be updated to the screen. 
"""

def config() -> argparse.Namespace:
    from desktop_env.desktop_env import DesktopEnv

    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default="vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx")
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--client_password", type=str, default="password") # osworld-public-evaluation for aws
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")

    # agent config
    parser.add_argument("--oai_config_path", type=str, default="mm_agents/coact/OAI_CONFIG_LIST")
    parser.add_argument("--orchestrator_model", type=str, default="o3-2025-04-16")
    parser.add_argument("--coding_model", type=str, default="o4-mini-2025-04-16")
    parser.add_argument("--summarizer_model", type=str, default="o4-mini-2025-04-16")
    parser.add_argument("--cua_model", type=str, default="computer-use-preview")
    parser.add_argument("--orchestrator_max_steps", type=int, default=15) #15
    parser.add_argument("--coding_max_steps", type=int, default=20) #20
    parser.add_argument("--cua_max_steps", type=int, default=25) #25
    parser.add_argument("--cut_off_steps", type=int, default=50) #200

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default=os.path.join('evaluation_examples', 'test_one.json')
    )
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples/examples"
    )
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores")

    # RAG related
    parser.add_argument("--summarize_rag", action='store_true', help="Whether to summarize RAG for the agent")
    parser.add_argument("--rag", action='store_true', help="Whether to use RAG for the agent")
    parser.add_argument("--rag_topk", type=int, default=4, help="Top k to use for RAG")
    parser.add_argument("--rag_filename", type=str, default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt", help="RAG retrieved context file name")

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results/coact_15_10_10_20")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")

    args = parser.parse_args()
    
    args.env = DesktopEnv(
        path_to_vm=args.path_to_vm,
        snapshot_name=args.snapshot_name,
        headless=args.headless,
        require_a11y_tree=False
    )
    return args

def process_task(task_info, 
                env,
                path_to_vm,
                snapshot_name="init_state",
                orchestrator_model="o3",
                coding_model='o4-mini',
                summarizer_model='o4-mini',
                cua_model='computer-use-preview',
                result_dir='results/coact',
                orchestrator_max_steps=15,
                cua_max_steps=25,
                coding_max_steps=20,
                cut_off_steps=50,
                screen_width=1920,
                screen_height=1080,
                sleep_after_execution=0.5,
                config_path="OAI_CONFIG_LIST",
                client_password="",
                summarize_rag=False,
                rag=False,
                rag_topk=4,
                rag_filename="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt",
                headless=False,
                logger=None   
                ):
    """Worker function to process a single task"""
    domain, ex_id, cfg = task_info
    
    # Record start time for execution time tracking
    start_time = time.time()
    
    # Recreate llm_config inside the worker process
    llm_config = LLMConfig.from_json(path=config_path).where(model=orchestrator_model)
    
    history_save_dir = os.path.join(result_dir, f"{domain}/{ex_id}")
    task_config = json.load(open(cfg))

    # Build context using the common function
    example_dir = os.path.dirname(cfg)
    additional_context = build_additional_contexts(
        example_dir=example_dir,
        use_rag=rag,
        summarize_rag=summarize_rag,
        rag_topk=rag_topk,  
        rag_filename=rag_filename
    )

    logger.info(f"[Domain]: {domain}")
    logger.info(f"[Example ID]: {ex_id}")
    instruction = task_config['instruction'] + additional_context
    logger.info(f"[Instruction]: {instruction}")

    try:
        with llm_config:
            orchestrator = OrchestratorAgent(
                name="orchestrator",
                system_message=TASK_DESCRIPTION
            )
            orchestrator_proxy = OrchestratorUserProxyAgent(
                env,
                name="orchestrator_proxy",
                is_termination_msg=lambda x: x.get("content", "") and (x.get("content", "")[0]["text"].lower() == "terminate" or x.get("content", "")[0]["text"].lower() == "infeasible"),
                human_input_mode="NEVER",
                path_to_vm=path_to_vm,
                snapshot_name=snapshot_name,
                screen_width=screen_width,
                screen_height=screen_height,
                sleep_after_execution=sleep_after_execution,
                code_execution_config=False,
                history_save_dir=history_save_dir,
                coding_model=coding_model,
                summarizer_model=summarizer_model,
                cua_model=cua_model,
                truncate_history_inputs=cua_max_steps + 1,
                cua_max_steps=cua_max_steps,
                coding_max_steps=coding_max_steps,
                cut_off_steps=cut_off_steps,
                client_password=client_password,
                user_instruction=task_config["instruction"],
                headless=headless,
                config_path=config_path
            )

        orchestrator_proxy.reset(task_config=task_config)
        time.sleep(60)
        screenshot = orchestrator_proxy.env.controller.get_screenshot()

        with open(os.path.join(history_save_dir, f'initial_screenshot_orchestrator.png'), "wb") as f:
            f.write(screenshot)
            
        # Prepare the initial message with optional RAG context
        initial_message = instruction + '\n\nCheck my computer screenshot and describe it first. If this task is possible to complete, please complete it on my computer. If not, reply with "INFEASIBLE" to end the conversation.\nI will not provide further information to you.'

        initial_message += "<img data:image/png;base64," + base64.b64encode(screenshot).decode("utf-8") + ">"
        
        orchestrator_proxy.initiate_chat(
            recipient=orchestrator,
            message=initial_message,
            max_turns=orchestrator_max_steps
        )
        
        chat_history = []
        key = list(orchestrator_proxy.chat_messages.keys())[0]
        chat_messages = orchestrator_proxy.chat_messages[key]
        for item in chat_messages:
            item.pop('tool_responses', None)
            if item.get('role', None) in ['tool', 'assistant'] and item.get('content', None):
                for msg in item['content']:
                    if msg.get('type', None) == 'image_url':
                        msg['image_url'] = "<image>"
            chat_history.append(item)
        
        # with open(os.path.join(history_save_dir, f'chat_history.json'), "w") as f:
        #     json.dump(chat_history, f)

        # Check if last message indicates task is infeasible (with safe access)
        if chat_history:
            last_msg = chat_history[-1]
            if last_msg.get('role') == 'user':
                content = last_msg.get('content', [])
                if isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict) and 'INFEASIBLE' in first_item.get('text', ''):
                        orchestrator_proxy.env.action_history.append("FAIL")

        cua_steps = len(glob.glob(f"{history_save_dir}/cua_output*/step_*.png"))
        coding_paths = glob.glob(f"{history_save_dir}/coding_output*/chat_history.json")
        coding_steps = 0
        for hist in coding_paths:
            with open(hist, 'r') as f:
                hist = json.dumps(json.load(f))
                coding_steps += hist.count('exitcode:')
        
        score = orchestrator_proxy.env.evaluate()
        
        orchestrator_usage = orchestrator.get_total_usage().get(orchestrator_model)
        orchestrator_prompt_tokens = orchestrator_usage.get('prompt_tokens')
        orchestrator_completion_tokens = orchestrator_usage.get('completion_tokens')
        
        # Count actual number of images sent to orchestrator (initial + GUI agent returns)
        orchestrator_image_count = count_images_in_messages(chat_history)
        
        prompt_price, completion_price = get_price(orchestrator_model)
        orchestrator_cost = orchestrator_prompt_tokens * prompt_price + orchestrator_completion_tokens * completion_price

        model_usage = orchestrator_proxy.model_usage
        model_usage["orchestrator"] = {
            "model_name": orchestrator_model,
            "cost": orchestrator_cost,
            "prompt_tokens": orchestrator_prompt_tokens,
            "completion_tokens": orchestrator_completion_tokens,
            "image_count": orchestrator_image_count
        }
        
        # Add model_name to existing model_usage entries
        if "cua" in model_usage:
            model_usage["cua"]["model_name"] = cua_model
        if "coding" in model_usage:
            model_usage["coding"]["model_name"] = coding_model
        if "summarizer" in model_usage:
            model_usage["summarizer"]["model_name"] = summarizer_model
        
        # Combine token usage from both agents
        prompt_tokens = sum(model_usage[model]["prompt_tokens"] for model in model_usage)
        completion_tokens = sum(model_usage[model]["completion_tokens"] for model in model_usage)
        total_cost = sum(model_usage[model]["cost"] for model in model_usage)
        total_image_count = sum(model_usage[model]["image_count"] for model in model_usage)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Merge conversation history and action logs into unified timeline
        unified_timeline = merge_conversation_and_actions(chat_history, orchestrator_proxy.action_logs)
        
        execution_log = {
            "statistics": {
                "score": score,
                "total_steps": cua_steps + coding_steps,
                "cua_steps": cua_steps,
                "coding_steps": coding_steps,
                "image_count": total_image_count,
                "total_cost": total_cost,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "execution_time": execution_time,
                "model_usage": model_usage
            },
            "task_config": task_config,
            "additional_context": additional_context,
            "action_logs": unified_timeline
        }
        
        # Save unified execution log
        with open(os.path.join(history_save_dir, "execution_log.json"), "w") as f:
            json.dump(serialize_json(execution_log), f, indent=2)
        
        logger.info(f"Score: {score}")
        with open(os.path.join(history_save_dir, f'result.txt'), "w") as f:
            f.write(str(score))
        
        if orchestrator_proxy.env is not None:
            orchestrator_proxy.env.close()
                
    except Exception as e:
        logger.error(f"Error processing task {domain}/{ex_id}")
        logger.error(traceback.format_exc())
        score = 0.0
        with open(os.path.join(history_save_dir, f'result.txt'), "w") as f:
            f.write(str(score))
        with open(os.path.join(history_save_dir, f'err_reason.txt'), "w") as f:
            f.write(f"Fatal error: {str(e)}\n\n{traceback.format_exc()}")
        
        # Skip saving execution_log when error occurs (err_reason.txt already saved)
        logger.info(f"Task failed with error, err_reason.txt saved, skipping execution_log.json")
        
        # Try to close environment
        try:
            if 'orchestrator_proxy' in locals() and orchestrator_proxy is not None:
                if orchestrator_proxy.env is not None:
                    orchestrator_proxy.env.close()
        except:
            pass
    
    return domain, score


def run(args, logger=None, tasks=None):
    """
    Run evaluation tasks.
    
    Args:
        args: Command line arguments
        logger: Logger instance (optional, will create if not provided)
        tasks: List of (domain, task_id) tuples (optional, will build from file if not provided)
    """
    result_name = os.path.basename(args.result_dir)

    if logger is None:
        logger = setup_logger(result_name, args.log_level)

    # Build tasks if not provided
    if tasks is None:
        with open(args.test_all_meta_path, encoding="utf-8") as f:
            test_all_meta = json.load(f)
        
        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}
        
        tasks = []
        for domain in test_all_meta:
            for ex_id in test_all_meta[domain]:
                tasks.append((domain, ex_id))
    
    if not args.get_score:
        scores: Dict[str, List[float]] = {}
        
        save_args_to_settings(args)

        # Execute all tasks
        if not tasks:
            logger.info("No tasks to process.")
        else:
            logger.info(f"Processing {len(tasks)} tasks...")

            # Process tasks sequentially
            results = []
            for domain, ex_id in tqdm(tasks, desc="Processing tasks"):
                # Prepare task directory and config
                target_dir = os.path.join(args.result_dir, f"{domain}/{ex_id}")
                cfg_path = os.path.join(args.test_config_base_dir, f"{domain}/{ex_id}/{ex_id}.json")
                if not os.path.exists(cfg_path):
                    cfg_path = os.path.join(args.test_config_base_dir, f"{domain}/{ex_id}.json")
                
                # Clean up existing directory and prepare for execution
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                os.makedirs(target_dir, exist_ok=True)
                
                # Create task tuple for process_task
                task = (domain, ex_id, cfg_path)
                result = process_task(task,
                                env=args.env,
                                path_to_vm=args.path_to_vm,
                                snapshot_name=args.snapshot_name,
                                result_dir=args.result_dir,
                                coding_model=args.coding_model,
                                summarizer_model=args.summarizer_model,
                                cua_model=args.cua_model,
                                orchestrator_model=args.orchestrator_model,
                                config_path=args.oai_config_path, 
                                orchestrator_max_steps=args.orchestrator_max_steps,
                                cua_max_steps=args.cua_max_steps,
                                coding_max_steps=args.coding_max_steps,
                                cut_off_steps=args.cut_off_steps,
                                screen_width=args.screen_width,
                                screen_height=args.screen_height,
                                sleep_after_execution=args.sleep_after_execution,
                                client_password=args.client_password,
                                summarize_rag=args.summarize_rag,
                                rag=args.rag,
                                rag_topk=args.rag_topk,
                                rag_filename=args.rag_filename,
                                headless=args.headless,
                                logger=logger)
                results.append(result)

            # Collect scores from results
            for domain, score in results:
                if domain not in scores:
                    scores[domain] = []
                scores[domain].append(score)

            # Print summary
            logger.info("\n=== Task Processing Complete ===")
            for domain in scores:
                if scores[domain]:
                    avg_score = sum(scores[domain]) / len(scores[domain])
                    logger.info(f"{domain}: {len(scores[domain])} tasks, average score: {avg_score:.2%}")
    
    # Build test_all_meta from tasks for summary
    test_all_meta = {}
    for domain, task_id in (tasks if tasks else []):
        if domain not in test_all_meta:
            test_all_meta[domain] = []
        test_all_meta[domain].append(task_id)
    
if __name__ == "__main__":
    args = config()
    run(args)
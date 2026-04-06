import datetime
import json
import logging
import os
import time
from typing import *
from wrapt_timeout_decorator import *
from agents.utils import serialize_json

logger = logging.getLogger("desktopenv.experiment")


def run_single_example(
    agent,
    env,
    example,
    max_steps,
    instruction,
    additional_context,
    args,
    example_result_dir,
    scores,
):
    method = args.method
    agent.reset()

    env.reset(task_config=example)
    # time.sleep(60)  # Wait for the environment to be ready
    obs = env._get_obs()  # Get the initial observation

    save_dir = os.path.join(example_result_dir, "operations")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"step_0.png"), "wb") as _f:
        _f.write(obs["screenshot"])

    # with open(
    #     os.path.join(example_result_dir, "instruction.txt"), "w", encoding="utf-8"
    # ) as f:
    #     f.write(instruction)

    done = False
    step_idx = 0
    action_logs = []  # Collect action logs for execution_log
    cua_steps = 0  # Count CUA/GUI steps
    coding_steps = 0  # Count coding agent steps
    start_time = time.time()  # Record start time

    # Helper function to get current token usage snapshot
    def get_usage_snapshot():
        """Get current token usage from all agent components."""
        usage = {}

        if method == "gta1":
            # GTA1Agent: Get stats from model_usage dictionary
            for agent_name, stats in agent.model_usage.items():
                usage[agent_name] = {
                    "prompt_tokens": stats.get("prompt_tokens", 0),
                    "completion_tokens": stats.get("completion_tokens", 0),
                    "image_count": stats.get("image_count", 0),
                    "cost": stats.get("cost", 0.0),
                }

        elif method == "agents2":
            # agents2: Get Manager (planner) stats
            if hasattr(agent, 'planner') and hasattr(agent.planner, 'get_usage_stats'):
                manager_stats = agent.planner.get_usage_stats()
                for agent_name in ['generator_agent', 'dag_translator_agent',
                                  'narrative_summarization_agent', 'episode_summarization_agent']:
                    if agent_name in manager_stats:
                        stats = manager_stats[agent_name]

                        # Calculate cost using engine's get_cost method
                        cost = 0.0
                        agent_obj = getattr(agent.planner, agent_name, None)
                        if agent_obj and hasattr(agent_obj, 'engine') and hasattr(agent_obj.engine, 'get_cost'):
                            cost = agent_obj.engine.get_cost()

                        usage[f"manager_{agent_name}"] = {
                            "prompt_tokens": stats.prompt_tokens,
                            "completion_tokens": stats.completion_tokens,
                            "image_count": stats.image_count,
                            "cost": cost,
                        }

        if method in ["agents2", "agents3"]:
            # Get Worker (executor) stats
            if hasattr(agent, 'executor') and hasattr(agent.executor, 'get_usage_stats'):
                worker_stats = agent.executor.get_usage_stats()
                for agent_name in ['generator_agent', 'reflection_agent']:
                    if agent_name in worker_stats:
                        stats = worker_stats[agent_name]

                        # Calculate cost using engine's get_cost method
                        cost = 0.0
                        agent_obj = getattr(agent.executor, agent_name, None)
                        if agent_obj and hasattr(agent_obj, 'engine') and hasattr(agent_obj.engine, 'get_cost'):
                            cost = agent_obj.engine.get_cost()

                        usage[f"worker_{agent_name}"] = {
                            "prompt_tokens": stats.prompt_tokens,
                            "completion_tokens": stats.completion_tokens,
                            "image_count": stats.image_count,
                            "cost": cost,
                        }

            # Get grounding agent stats
            if hasattr(agent, 'executor') and hasattr(agent.executor, 'grounding_agent'):
                grounding_llm = agent.executor.grounding_agent.grounding_model
                if hasattr(grounding_llm, 'engine') and hasattr(grounding_llm.engine, 'get_usage_stats'):
                    stats = grounding_llm.engine.get_usage_stats()

                    # Calculate cost using engine's get_cost method
                    cost = 0.0
                    if hasattr(grounding_llm.engine, 'get_cost'):
                        cost = grounding_llm.engine.get_cost()

                    usage["grounding_agent"] = {
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "image_count": stats.image_count,
                        "cost": cost,
                    }

        return usage

    # Helper function to calculate usage delta
    def calculate_usage_delta(before, after):
        """Calculate the difference in token usage between two snapshots."""
        delta = {}
        all_keys = set(before.keys()) | set(after.keys())

        for key in all_keys:
            before_stats = before.get(key, {"prompt_tokens": 0, "completion_tokens": 0, "image_count": 0, "cost": 0.0})
            after_stats = after.get(key, {"prompt_tokens": 0, "completion_tokens": 0, "image_count": 0, "cost": 0.0})

            delta[key] = {
                "prompt_tokens": after_stats["prompt_tokens"] - before_stats["prompt_tokens"],
                "completion_tokens": after_stats["completion_tokens"] - before_stats["completion_tokens"],
                "image_count": after_stats["image_count"] - before_stats["image_count"],
                "cost": after_stats["cost"] - before_stats["cost"],
            }

        # Calculate total
        total = {"prompt_tokens": 0, "completion_tokens": 0, "image_count": 0, "cost": 0.0}
        for stats in delta.values():
            total["prompt_tokens"] += stats["prompt_tokens"]
            total["completion_tokens"] += stats["completion_tokens"]
            total["image_count"] += stats["image_count"]
            total["cost"] += stats["cost"]

        delta["total"] = total
        return delta

    try:
        if args.record:
            env.controller.start_recording()
        while not done and step_idx < max_steps:
            # Record step start time
            step_start_time = time.time()

            # Get token usage snapshot before this step
            usage_before_step = get_usage_snapshot()

            # Set current step info for code agent budget tracking
            if hasattr(agent, "grounding_agent") and hasattr(
                agent.grounding_agent, "set_step_info"
            ):
                agent.grounding_agent.set_step_info(step_idx, max_steps)

            response, actions = agent.predict(instruction + additional_context, obs)

            # Get token usage snapshot after prediction
            usage_after_step = get_usage_snapshot()

            # Calculate step execution time
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            # Calculate token usage for this step
            step_token_usage = calculate_usage_delta(usage_before_step, usage_after_step)

            # Check if code agent was called and account for its steps
            code_agent_steps = 0
            action_type = "gui_action"  # Default type

            if response.get("code_agent_output"):
                code_agent_steps = response["code_agent_output"].get(
                    "steps_executed", 0
                )
                action_type = "code_action"
                coding_steps += code_agent_steps
                logger.info(f"Code agent executed {code_agent_steps} internal steps")
                logger.info(
                    f"These {code_agent_steps} steps will be counted toward max_steps={max_steps}"
                )

            for action in actions:
                # Determine action type based on action content
                if (
                    action_type != "code_action"
                ):  # Only check if not already identified as code_action
                    action_stripped = action.strip()
                    if action_stripped == "DONE":
                        action_type = "done_action"
                    elif action_stripped == "FAIL":
                        action_type = "fail_action"
                    elif action_stripped == "WAIT" or "time.sleep" in action:
                        action_type = "wait_action"
                    else:
                        action_type = "gui_action"
                        cua_steps += 1

                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                logger.info("Step %d (%s): %s", step_idx + 1, action_type, action)
                obs, reward, done, info = env.step(action, args.sleep_after_execution)

                logger.info("Reward: %.2f", reward)
                logger.info("Done: %s", done)
                # Save screenshot and trajectory information
                with open(os.path.join(save_dir, f"step_{step_idx + 1}.png"), "wb") as _f:
                    _f.write(obs["screenshot"])

                response.update(
                    {
                        "step_num": step_idx + 1,
                        "action_timestamp": action_timestamp,
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "info": info,
                    }
                )

                # Collect action log for execution_log
                action_log = {
                    "step": step_idx + 1,
                    "type": action_type,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "screenshot": os.path.join(save_dir, f"step_{step_idx + 1}.png"),
                    "timestamp": action_timestamp,
                    "step_time": round(step_time, 2),
                    "response": response.copy(),
                    "token_usage": step_token_usage,
                }
                action_logs.append(action_log)

                # with open(
                #     os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8"
                # ) as f:
                #     f.write(json.dumps(response, ensure_ascii=False))
                #     f.write("\n")
                if done:
                    logger.info("The episode is done.")
                    break

            # Increment step_idx by 1 (for the GUI action) plus code agent steps
            step_idx += 1 + code_agent_steps
            logger.info(f"Total steps used so far: {step_idx}/{max_steps}")

        logger.info(f"Episode ended. Total steps: {step_idx}/{max_steps}, Done: {done}")
        end_time = time.time()
        execution_time = end_time - start_time

        result = env.evaluate()
        logger.info("Result: %.2f", result)
        scores.append(result)

        # Get token usage statistics if available
        total_cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        image_count = 0
        model_usage = {}

        if method == "gta1":
            # GTA1Agent: Get stats from model_usage dictionary
            logger.info("Collecting GTA1Agent stats...")
            for agent_name, stats in agent.model_usage.items():
                # Determine actual model name
                if agent_name == "planner":
                    actual_model_name = getattr(agent, "planner_model", "unknown")
                elif agent_name == "judge":
                    actual_model_name = getattr(agent, "judge_model", "unknown")
                elif agent_name == "grounding":
                    actual_model_name = "gta1-7b"  # Default grounding model
                else:
                    actual_model_name = "unknown"
                
                model_usage[agent_name] = {
                    "model_name": actual_model_name,
                    "prompt_tokens": stats.get("prompt_tokens", 0),
                    "completion_tokens": stats.get("completion_tokens", 0),
                    "image_count": stats.get("image_count", 0),
                    "cost": stats.get("cost", 0.0),
                }
                total_cost += stats.get("cost", 0.0)
                prompt_tokens += stats.get("prompt_tokens", 0)
                completion_tokens += stats.get("completion_tokens", 0)
                image_count += stats.get("image_count", 0)

                logger.info(
                    f"GTA1 {agent_name} ({actual_model_name}) stats: prompt={stats.get('prompt_tokens', 0)}, "
                    f"completion={stats.get('completion_tokens', 0)}, images={stats.get('image_count', 0)}, "
                    f"cost=${stats.get('cost', 0.0):.4f}"
                )

        elif method == "agents2":
            # agents2: Get Manager (planner) stats
            logger.info("Collecting Manager stats...")
            if hasattr(agent.planner, 'get_usage_stats'):
                manager_stats = agent.planner.get_usage_stats()
                
                for agent_name in ['generator_agent', 'dag_translator_agent', 
                                  'narrative_summarization_agent', 'episode_summarization_agent']:
                    if agent_name in manager_stats:
                        stats = manager_stats[agent_name]
                        
                        # Get model name from agent object
                        agent_obj = getattr(agent.planner, agent_name, None)
                        actual_model_name = "unknown"
                        if agent_obj and hasattr(agent_obj, 'engine') and hasattr(agent_obj.engine, 'model'):
                            actual_model_name = agent_obj.engine.model
                        
                        model_usage[agent_name] = {
                            "model_name": actual_model_name,
                            "prompt_tokens": stats.prompt_tokens,
                            "completion_tokens": stats.completion_tokens,
                            "image_count": stats.image_count,
                            "cost": 0.0,
                        }
                        prompt_tokens += stats.prompt_tokens
                        completion_tokens += stats.completion_tokens
                        image_count += stats.image_count
                        
                        # Calculate cost if available
                        if agent_obj and hasattr(agent_obj, 'engine') and hasattr(agent_obj.engine, 'get_cost'):
                            cost = agent_obj.engine.get_cost()
                            model_usage[agent_name]["cost"] = cost
                            total_cost += cost
                        
                        logger.info(
                            f"Manager {agent_name} ({actual_model_name}) stats: prompt={stats.prompt_tokens}, "
                            f"completion={stats.completion_tokens}, images={stats.image_count}, "
                            f"cost=${model_usage[agent_name]['cost']:.4f}"
                        )

        if method in ["agents2", "agents3"]:
            # Get Worker (executor) stats - both agents2 and agents3 have this
            logger.info("Collecting Worker stats...")
            if hasattr(agent, 'executor') and hasattr(agent.executor, 'get_usage_stats'):
                worker_stats = agent.executor.get_usage_stats()

                for agent_name in ['generator_agent', 'reflection_agent']:
                    if agent_name in worker_stats:
                        stats = worker_stats[agent_name]
                        
                        # Get model name from agent object
                        agent_obj = getattr(agent.executor, agent_name, None)
                        actual_model_name = "unknown"
                        if agent_obj and hasattr(agent_obj, 'engine') and hasattr(agent_obj.engine, 'model'):
                            actual_model_name = agent_obj.engine.model
                        
                        model_usage[agent_name] = {
                            "model_name": actual_model_name,
                            "prompt_tokens": stats.prompt_tokens,
                            "completion_tokens": stats.completion_tokens,
                            "image_count": stats.image_count,
                            "cost": 0.0,
                        }
                        prompt_tokens += stats.prompt_tokens
                        completion_tokens += stats.completion_tokens
                        image_count += stats.image_count

                        # Calculate cost if available
                        if agent_obj and hasattr(agent_obj, 'engine') and hasattr(agent_obj.engine, 'get_cost'):
                            cost = agent_obj.engine.get_cost()
                            model_usage[agent_name]["cost"] = cost
                            total_cost += cost

                        logger.info(
                            f"{'Worker ' if method == 'agents2' else ''}{agent_name} ({actual_model_name}) stats: prompt={stats.prompt_tokens}, "
                            f"completion={stats.completion_tokens}, images={stats.image_count}, "
                            f"cost=${model_usage[agent_name]['cost']:.4f}"
                        )

            # Get grounding agent stats - both agents2 and agents3 have this
            logger.info("Collecting Grounding agent stats...")
            if hasattr(agent.executor, 'grounding_agent'):
                grounding_llm = agent.executor.grounding_agent.grounding_model
                if hasattr(grounding_llm, 'engine') and hasattr(grounding_llm.engine, 'get_usage_stats'):
                    stats = grounding_llm.engine.get_usage_stats()
                    
                    # Get model name from grounding agent
                    actual_model_name = "unknown"
                    if hasattr(grounding_llm.engine, 'model'):
                        actual_model_name = grounding_llm.engine.model
                    
                    model_usage["grounding_agent"] = {
                        "model_name": actual_model_name,
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "image_count": stats.image_count,
                        "cost": 0.0,
                    }
                    prompt_tokens += stats.prompt_tokens
                    completion_tokens += stats.completion_tokens
                    image_count += stats.image_count

                    if hasattr(grounding_llm.engine, 'get_cost'):
                        cost = grounding_llm.engine.get_cost()
                        model_usage["grounding_agent"]["cost"] = cost
                        total_cost += cost

                    logger.info(
                        f"Grounding agent ({actual_model_name}) stats: prompt={stats.prompt_tokens}, "
                        f"completion={stats.completion_tokens}, images={stats.image_count}, "
                        f"cost=${model_usage['grounding_agent']['cost']:.4f}"
                    )

        logger.info(
            f"Total usage stats: prompt={prompt_tokens}, completion={completion_tokens}, "
            f"images={image_count}, cost=${total_cost:.4f}"
        )

        # Create execution_log with structure matching langgraph_agent.py (normal completion)
        execution_log = {
            "statistics": {
                "score": result,
                "total_steps": step_idx,
                "cua_steps": cua_steps,
                "coding_steps": coding_steps,
                "total_cost": total_cost,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "image_count": image_count,
                "execution_time": execution_time,
                "model_usage": model_usage,
            },
            "task_config": example,
            "additional_context": additional_context,
            "action_logs": action_logs,
        }

        # Save result.txt
        with open(
            os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(f"{result}\n")

        if args.record:
            env.controller.end_recording(
                os.path.join(example_result_dir, "recording.mp4")
            )
        
        # Save execution_log.json
        with open(
            os.path.join(example_result_dir, "execution_log.json"),
            "w",
            encoding="utf-8",
        ) as f:
            # Make execution_log JSON-serializable before saving
            serializable_log = serialize_json(execution_log)
            json.dump(serializable_log, f, indent=2, ensure_ascii=False)

        logger.info(f"Execution completed in {execution_time:.2f}s")
        logger.info(f"Total steps: {step_idx}")
        logger.info(f"  - CUA/GUI steps: {cua_steps}")
        logger.info(f"  - Coding steps: {coding_steps}")
        logger.info(
            f"Total cost: ${total_cost:.4f}, Tokens: {prompt_tokens} prompt + {completion_tokens} completion"
        )
        logger.info("Execution log saved to execution_log.json")

    except Exception as e:
        logger.error(f"Execution error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        
        # Calculate execution time even for errors
        execution_time = time.time() - start_time

        # Create emergency execution_log (matching langgraph_agent.py structure)
        execution_log = {
            "statistics": {
                "score": 0.0,
                "total_steps": step_idx,
                "cua_steps": cua_steps,
                "coding_steps": coding_steps,
                "total_cost": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "image_count": 0,
                "execution_time": execution_time,
                "model_usage": {},
            },
            "task_config": example,
            "additional_context": additional_context,
            "action_logs": action_logs,
            "error": traceback.format_exc(),
        }

        with open(
            os.path.join(example_result_dir, "execution_log.json"),
            "w",
            encoding="utf-8",
        ) as f:
            # Make execution_log JSON-serializable before saving
            serializable_log = serialize_json(execution_log)
            json.dump(serializable_log, f, indent=2, ensure_ascii=False)

        with open(
            os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("0.0\n")

        scores.append(0.0)
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import base64
import glob
import json
import os
import traceback
from typing import Any, Callable, Literal, Optional, Union

from .autogen.llm_config import LLMConfig
from .autogen.agentchat.agent import Agent
from .autogen.agentchat.conversable_agent import ConversableAgent
from .autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

from .cua_agent import run_cua
from .coding_agent import TerminalProxyAgent, CODER_SYSTEM_MESSAGE
from agents.utils import get_price

ONLY_CUA = False  # False update

class OrchestratorAgent(MultimodalConversableAgent):
    """(In preview) Captain agent, designed to solve a task with an agent or a group of agents."""

    CALL_GUI_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "call_gui_agent",
            "description": """Let a OS Operator to solve a task. OS operator can operate the computer by clicking and typing (not accurate in dense UI). Require detailed task description.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "[REQUIRED] A detailed task to be solved with step-by-step guidance.",
                    },
                },
            },
        },
    }

    CALL_CODING_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "call_coding_agent",
            "description": """(You MUST use this first) Let a programmer to solve a task. Coding agent can write python and bash code with many tools to solve a task. Require detailed task and environment description.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "[REQUIRED] A detailed task to be solved.",
                    },
                    "environment": {
                        "type": "string",
                        "description": "[REQUIRED] The environment description of the coding agent. It should be a detailed description of the system state, including the opened files, the running processes, etc.",
                    }
                },
            },
        },
    }

    CALL_API_SUMMARY_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "call_api_summary_agent",
            "description": """Let a API summary agent to summarize the API response. API summary agent can summarize the API response. Require detailed API response.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "[REQUIRED] A url of the API response."},
                },
            },
        },
    }

    DEFAULT_DESCRIPTION = ""

    # This is used to prompt the LLM to summarize the conversation history between CaptainAgent's tool execution history
    DEFAULT_SUMMARY_PROMPT = "Read the following conversation history between an expert and a group of agent experts, summarize the conversation history. Your summarization should include the initial task, the experts' plan and the attempt, finally the results of the conversation. If the experts arrived at a conclusion, state it as it is without any modification."

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = None,
        llm_config: Optional[Union[LLMConfig, dict[str, Any], Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict[str, Any], Literal[False]]] = False,
        description: Optional[str] = DEFAULT_DESCRIPTION,
        **kwargs: Any,
    ):
        super().__init__(
            name,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            description=description,
            **kwargs,
        )

        if system_message is None:
            self.update_system_message("")
        else:
            self.update_system_message(system_message)

        if not ONLY_CUA:
            self.update_tool_signature(self.CALL_CODING_AGENT_TOOL, is_remove=False)
        self.update_tool_signature(self.CALL_GUI_AGENT_TOOL, is_remove=False)
        # self.assistant.update_tool_signature(self.CALL_API_SUMMARY_AGENT_TOOL, is_remove=False)  # TODO: add this tool later


class OrchestratorUserProxyAgent(MultimodalConversableAgent):
    """(In preview) A proxy agent for the captain agent, that can execute code and provide feedback to the other agents."""

    DEFAULT_AUTO_REPLY = "Thank you! Note that the user's task is: {user_instruction}. Please continue the task. If you think the everything is solved, please reply me only with 'TERMINATE'. But once you think the task is impossible to solve, please reply me only with 'INFEASIBLE'."

    DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
        "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
        "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
        "NEVER": "A computer terminal that can running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks), or the conversation history and result of a group of agents",
    }

    CONVERSATION_REVIEW_PROMPT = """You are looking for a conversation history between a user and an agent.
    Given the conversation history below, summarize the conversation history in a concise way.

    - Conversation history:
    {chat_history}

    - Response template (markdown format):
    # Summarize of the conversation history
    ...(include the middle terminal output. They are important.)

    # Final result
    ...
    """

    def __init__(
        self,
        env, 
        name: str,
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict[str, Any], Literal[False]]] = {},
        default_auto_reply: Optional[Union[str, dict[str, Any]]] = DEFAULT_AUTO_REPLY,
        llm_config: Optional[Union[LLMConfig, dict[str, Any], Literal[False]]] = False,
        system_message: Optional[Union[str, list]] = "",
        description: Optional[str] = None,

        # GUI Agent config
        path_to_vm: str = None,
        snapshot_name: str = "init_state",
        observation_type: str = "screenshot",
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 1.0,
        truncate_history_inputs: int = 51,
        cua_max_steps: int = 50,
        coding_max_steps: int = 30,
        cut_off_steps: int = 200,
        history_save_dir: str = "",
        coding_model: str = "o4-mini-2025-04-16",
        summarizer_model: str = "o4-mini-2025-04-16",
        cua_model: str = "computer-use-preview",
        client_password: str = "",
        user_instruction: str = "",
        headless: bool = False,
        config_path: str = "coact/OAI_CONFIG_LIST",
    ):
        description = (
            description if description is not None else self.DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS[human_input_mode]
        )
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply.format(user_instruction=user_instruction),
            description=description,
        )
        if ONLY_CUA:
            self.register_function(
                function_map={
                    "call_gui_agent": lambda **args: self._call_gui_agent(**args, screen_width=screen_width, screen_height=screen_height),
                }
            )
        else:
            self.register_function(
                function_map={
                    "call_gui_agent": lambda **args: self._call_gui_agent(**args, screen_width=screen_width, screen_height=screen_height),
                    "call_coding_agent": lambda **args: self._call_coding_agent(**args),
                }
            )
        self._code_execution_config = code_execution_config
        self.cua_config = {
            "max_steps": cua_max_steps,
            "sleep_after_execution": sleep_after_execution,
            "truncate_history_inputs": truncate_history_inputs,
        }
        self.client_password = client_password

        self.env = env

        self.history_save_dir = history_save_dir
        self.cua_call_count = 0
        self.coding_call_count = 0
        self.cua_max_steps = cua_max_steps
        self.coding_max_steps = coding_max_steps
        self.cut_off_steps = cut_off_steps
        # self.llm_config = llm_config
        self.coding_model = coding_model
        self.summarizer_model = summarizer_model
        self.cua_model = cua_model
        llm_configs = json.load(open(config_path, "r"))
        self.llm_config = next((config for config in llm_configs if config["model"] == coding_model), None)

        # Add statistics tracking
        self.action_logs = []  # Unified action log list
        self.global_step = 0  # Global step counter across all agent calls

        # Track usage by model
        self.model_usage = {}

        # Track last orchestrator usage for incremental tracking
        self.last_orchestrator_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    def reset(self, task_config: dict[str, Any]):
        obs = self.env.reset(task_config=task_config)
        return obs

    def generate_reply(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Optional[Union[str, dict[str, Any]]]:
        
        current_steps = self._count_current_steps()
        if current_steps >= self.cut_off_steps:
            print(f"Reached cut_off_steps limit: {current_steps}/{self.cut_off_steps}")
            return {
                "role": "assistant", 
                "content": [{"type": "text", "text": "TERMINATE"}]
            }
        
        return super().generate_reply(messages=messages, sender=sender, **kwargs)
    
    def _count_current_steps(self) -> int:
        cua_steps = len(glob.glob(f"{self.history_save_dir}/cua_output*/step_*.png"))
        
        coding_paths = glob.glob(f"{self.history_save_dir}/coding_output*/chat_history.json")
        coding_steps = 0
        for hist_path in coding_paths:
            try:
                with open(hist_path, 'r') as f:
                    hist_content = json.dumps(json.load(f))
                    coding_steps += hist_content.count('exitcode:')
            except:
                pass
        
        return cua_steps + coding_steps

    def _call_gui_agent(self, task: str, screen_width: int = 1920, screen_height: int = 1080) -> str:
        """Run a GUI agent to solve the task."""
        import time

        # Record start time for this GUI agent call
        gui_agent_start_time = time.time()

        cua_path = os.path.join(self.history_save_dir, f'cua_output_{self.cua_call_count}')
        if not os.path.exists(cua_path):
            os.makedirs(cua_path)

        # Calculate remaining step budget based on cut_off_steps
        current_steps = self._count_current_steps()
        remaining_steps = self.cut_off_steps - current_steps
        # Use the minimum of remaining budget and cua_max_steps
        actual_max_steps = min(self.cua_config["max_steps"], remaining_steps)

        if actual_max_steps <= 0:
            return "# Response from GUI agent: Reached step limit, cannot execute more steps."

        # Initialize variables to track usage even on failure
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        image_count = 0
        result = "ERROR"
        history_inputs = []

        try:
            history_inputs, result, cost, input_tokens, output_tokens, image_count, step_details = run_cua(self.env,
                                                   task,
                                                   save_path=cua_path,
                                                   max_steps=actual_max_steps,
                                                   sleep_after_execution=self.cua_config["sleep_after_execution"],
                                                   truncate_history_inputs=self.cua_config["truncate_history_inputs"],
                                                   client_password=self.client_password,
                                                   model=self.cua_model
                                                   )
            screenshot = self.env.controller.get_screenshot()

            with open(os.path.join(cua_path, "history_inputs.json"), "w") as f:
                json.dump(history_inputs, f)
            with open(os.path.join(cua_path, "result.txt"), "w") as f:
                f.write(result)
            with open(os.path.join(cua_path, "cost.txt"), "w") as f:
                f.write(str(cost))

            # Calculate total time for all GUI steps
            gui_agent_end_time = time.time()
            total_gui_time = gui_agent_end_time - gui_agent_start_time
            num_steps = len(step_details)
            time_per_step = total_gui_time / num_steps if num_steps > 0 else 0

            # Create action_log for each step
            for step_detail in step_details:
                # Increment global step counter
                self.global_step += 1

                # Build token_usage dict for this step
                token_usage = {
                    "cua": {
                        "prompt_tokens": step_detail["input_tokens"],
                        "completion_tokens": step_detail["output_tokens"],
                        "image_count": 1,  # Each step has one image
                        "cost": step_detail["cost"]
                    },
                    "total": {
                        "prompt_tokens": step_detail["input_tokens"],
                        "completion_tokens": step_detail["output_tokens"],
                        "image_count": 1,
                        "cost": step_detail["cost"]
                    }
                }

                # Record action log for each step (use global_step instead of call_index)
                action_log = {
                    "step": self.global_step,
                    "type": "gui_operator",
                    "task": task,
                    "result": step_detail.get("reasoning", ""),
                    "cost": step_detail["cost"],
                    "save_path": cua_path,
                    "model": self.cua_model,
                    "action": step_detail["actions"],  # List of actions for this step
                    "screenshot": os.path.join(cua_path, step_detail["screenshot"]),
                    "step_time": round(time_per_step, 2),
                    "token_usage": token_usage,
                    "error": step_detail.get("error")  # Include error if present
                }
                self.action_logs.append(action_log)

        except Exception as e:
            # Calculate time even for error case
            gui_agent_end_time = time.time()
            total_gui_time = gui_agent_end_time - gui_agent_start_time

            # Increment global step counter for error case
            self.global_step += 1

            # Build token_usage dict for error case
            token_usage = {
                "cua": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "image_count": image_count,
                    "cost": cost
                },
                "total": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "image_count": image_count,
                    "cost": cost
                }
            }

            # Record error in action log even when failed
            action_log = {
                "step": self.global_step,
                "type": "gui_operator",
                "task": task,
                "result": f"ERROR: {str(e)}",
                "cost": cost,
                "save_path": cua_path,
                "model": self.cua_model,
                "action": [],  # Empty scripts on error
                "step_time": round(total_gui_time, 2),
                "error": traceback.format_exc(),
                "token_usage": token_usage
            }
            self.action_logs.append(action_log)
            
        finally:
            # Always update model-specific usage regardless of success or failure
            if "cua" not in self.model_usage:
                self.model_usage["cua"] = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
            self.model_usage["cua"]["cost"] += cost
            self.model_usage["cua"]["prompt_tokens"] += input_tokens
            self.model_usage["cua"]["completion_tokens"] += output_tokens
            self.model_usage["cua"]["image_count"] += image_count
            
            self.cua_call_count += 1
        
        if "ERROR" in result:
            return f"# Response from GUI agent error: {result}"

        if "TERMINATE" in result:
            result = result.replace("TERMINATE", "").strip()
            if result == "":
                result = "Task completed. Please check the screenshot."
        elif "IDK" in result:
            result = result.replace("IDK", "").strip()
        else:
            result = f"I didn't complete the task and I have to go. Now I'm working on \"{result}\", please check the current screenshot."
        return f"# Response from GUI agent: {result}<img data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}>"
    
    def _call_coding_agent(self, task: str, environment: str) -> str:
        """Run a coding agent to solve the task."""
        import time

        # Record start time for this coding agent call
        coding_agent_start_time = time.time()

        default_auto_reply = "I'm a code interpreter and I can only execute your code or end the conversation. If you think the problem is solved, please reply me only with 'TERMINATE'."

        # Calculate remaining step budget based on cut_off_steps
        current_steps = self._count_current_steps()
        remaining_steps = self.cut_off_steps - current_steps
        # Use the minimum of remaining budget and coding_max_steps
        actual_max_steps = min(self.coding_max_steps, remaining_steps)

        if actual_max_steps <= 0:
            return "# Response from coding agent: Reached step limit, cannot execute more steps."

        # Initialize variables to track usage even on failure
        coding_agent = None
        summarizer = None
        code_interpreter = None
        summarized_history = "ERROR: Task execution failed"
        coding_output_path = os.path.join(self.history_save_dir, f'coding_output_{self.coding_call_count}')

        try:
            screenshot = self.env.controller.get_screenshot()
            coding_agent = MultimodalConversableAgent(
                name="coding_agent",
                llm_config=LLMConfig(api_type="openai", model=self.coding_model),
                system_message=CODER_SYSTEM_MESSAGE.format(CLIENT_PASSWORD=self.client_password),
            )
            summarizer = ConversableAgent(
                name="summarizer",
                llm_config=LLMConfig(api_type="openai", model=self.summarizer_model),
                system_message=self.CONVERSATION_REVIEW_PROMPT,
            )

            code_interpreter = TerminalProxyAgent(
                name="code_interpreter",
                human_input_mode="NEVER",
                code_execution_config={
                    "use_docker": False,
                    "timeout": 300,
                    "last_n_messages": 1,
                },
                max_consecutive_auto_reply = None,
                default_auto_reply = default_auto_reply,
                description = None,
                is_termination_msg=lambda x: x.get("content", "") and x.get("content", "")[0]["text"].lower() == "terminate",
                env=self.env,
            )

            code_interpreter.initiate_chat(
                recipient=coding_agent,
                message=f"# Task\n{task}\n\n# Environment\n{environment}<img data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}>",
                max_turns=actual_max_steps,
            )
        
            chat_history = []
            key = list(code_interpreter.chat_messages.keys())[0]
            chat_messages = code_interpreter.chat_messages[key]
            for item in chat_messages:
                for content in item['content']:
                    if content['type'] == 'image_url':
                        content['image_url']['url'] = '<image>'
                chat_history.append(item)
            
            if not os.path.exists(coding_output_path):
                os.makedirs(coding_output_path)
            
            with open(os.path.join(coding_output_path, "chat_history.json"), "w") as f:
                json.dump(chat_history, f)
            
            # Parse chat history to extract per-step information
            # Strategy: Find all messages with 'exitcode:' (execution results),
            # then look backward for the corresponding code
            step_infos = []
            step_no = 0

            for i, item in enumerate(chat_history):
                # Look for execution result messages (from code_interpreter)
                if item.get('role') == 'user':
                    content = item.get('content', '')
                    execution_result = None

                    # Extract execution result text
                    if isinstance(content, list):
                        for content_item in content:
                            if content_item.get('type') == 'text':
                                text = content_item.get('text', '')
                                if 'exitcode:' in text:
                                    execution_result = text
                                    break
                    elif isinstance(content, str) and 'exitcode:' in content:
                        execution_result = content

                    # If we found an execution result, look backward for the code
                    if execution_result:
                        step_no += 1
                        code_block = None

                        # Search backward through previous messages for the code
                        for j in range(i - 1, -1, -1):
                            prev_item = chat_history[j]
                            if prev_item.get('role') == 'assistant':
                                prev_content = prev_item.get('content', '')

                                # Extract code block
                                if isinstance(prev_content, list):
                                    for content_item in prev_content:
                                        if content_item.get('type') == 'text':
                                            text = content_item.get('text', '')
                                            # Try to extract code blocks
                                            if '```python' in text:
                                                start = text.find('```python') + 9
                                                end = text.find('```', start)
                                                if end > start:
                                                    code_block = text[start:end].strip()
                                                    break
                                            elif '```bash' in text:
                                                start = text.find('```bash') + 7
                                                end = text.find('```', start)
                                                if end > start:
                                                    code_block = text[start:end].strip()
                                                    break
                                            elif '```' in text:
                                                start = text.find('```') + 3
                                                end = text.find('```', start)
                                                if end > start:
                                                    code_block = text[start:end].strip()
                                                    break
                                elif isinstance(prev_content, str):
                                    if '```python' in prev_content:
                                        start = prev_content.find('```python') + 9
                                        end = prev_content.find('```', start)
                                        if end > start:
                                            code_block = prev_content[start:end].strip()
                                    elif '```bash' in prev_content:
                                        start = prev_content.find('```bash') + 7
                                        end = prev_content.find('```', start)
                                        if end > start:
                                            code_block = prev_content[start:end].strip()
                                    elif '```' in prev_content:
                                        start = prev_content.find('```') + 3
                                        end = prev_content.find('```', start)
                                        if end > start:
                                            code_block = prev_content[start:end].strip()

                                # If we found code, stop searching
                                if code_block:
                                    break

                        # Add step info (even if code_block is None, to preserve step count)
                        step_infos.append({
                            "step": step_no,
                            "code": code_block if code_block else "# Code not found",
                            "result": execution_result
                        })

            # Count coding steps (number of exitcode occurrences)
            coding_steps = len(step_infos)

            # Review the group chat history
            summarized_history = summarizer.generate_oai_reply(
                messages=[
                    {
                        "role": "user",
                        "content": self.CONVERSATION_REVIEW_PROMPT.format(chat_history=chat_history),
                    }
                ]
            )[1]

            # Get token usage before recording action log
            coding_prompt = 0
            coding_completion = 0
            coding_cost = 0.0
            summarizer_prompt = 0
            summarizer_completion = 0
            summarizer_cost = 0.0

            if coding_agent:
                coding_usage = coding_agent.get_total_usage()
                if self.coding_model in coding_usage:
                    coding_prompt = coding_usage[self.coding_model].get("prompt_tokens", 0)
                    coding_completion = coding_usage[self.coding_model].get("completion_tokens", 0)
                    prompt_price, completion_price = get_price(self.coding_model)
                    coding_cost = coding_prompt * prompt_price + coding_completion * completion_price

            if summarizer:
                summarizer_usage = summarizer.get_total_usage()
                if self.summarizer_model in summarizer_usage:
                    summarizer_prompt = summarizer_usage[self.summarizer_model].get("prompt_tokens", 0)
                    summarizer_completion = summarizer_usage[self.summarizer_model].get("completion_tokens", 0)
                    prompt_price, completion_price = get_price(self.summarizer_model)
                    summarizer_cost = summarizer_prompt * prompt_price + summarizer_completion * completion_price

            # Calculate total time and time per step
            coding_agent_end_time = time.time()
            total_coding_time = coding_agent_end_time - coding_agent_start_time

            # Create action_log for each coding step
            if coding_steps > 0:
                # Distribute coding agent token usage and time across steps
                coding_prompt_per_step = coding_prompt // coding_steps
                coding_completion_per_step = coding_completion // coding_steps
                coding_cost_per_step = coding_cost / coding_steps
                time_per_step = total_coding_time / coding_steps

                for i, step_info in enumerate(step_infos):
                    is_last_step = (i == len(step_infos) - 1)

                    # For the last step, include any remainder and the summarizer usage
                    if is_last_step:
                        step_coding_prompt = coding_prompt - (coding_prompt_per_step * (coding_steps - 1))
                        step_coding_completion = coding_completion - (coding_completion_per_step * (coding_steps - 1))
                        step_coding_cost = coding_cost - (coding_cost_per_step * (coding_steps - 1))
                        step_summarizer_prompt = summarizer_prompt
                        step_summarizer_completion = summarizer_completion
                        step_summarizer_cost = summarizer_cost
                    else:
                        step_coding_prompt = coding_prompt_per_step
                        step_coding_completion = coding_completion_per_step
                        step_coding_cost = coding_cost_per_step
                        step_summarizer_prompt = 0
                        step_summarizer_completion = 0
                        step_summarizer_cost = 0.0

                    # Increment global step counter
                    self.global_step += 1

                    # Build token_usage dict for this step
                    token_usage = {
                        "coding": {
                            "prompt_tokens": step_coding_prompt,
                            "completion_tokens": step_coding_completion,
                            "cost": step_coding_cost
                        },
                        "summarizer": {
                            "prompt_tokens": step_summarizer_prompt,
                            "completion_tokens": step_summarizer_completion,
                            "cost": step_summarizer_cost
                        },
                        "total": {
                            "prompt_tokens": step_coding_prompt + step_summarizer_prompt,
                            "completion_tokens": step_coding_completion + step_summarizer_completion,
                            "cost": step_coding_cost + step_summarizer_cost
                        }
                    }

                    # Record action log for this step (use global_step instead of call_index)
                    action_log = {
                        "step": self.global_step,
                        "type": "code_execution",
                        "task": task,
                        "environment": environment,
                        "model": self.coding_model,
                        "save_path": coding_output_path,
                        "code": step_info["code"],
                        "output": step_info["result"],
                        "step_time": round(time_per_step, 2),
                        "token_usage": token_usage
                    }
                    self.action_logs.append(action_log)
            else:
                # If no steps were executed, create a single action_log with all usage
                # Increment global step counter
                self.global_step += 1

                token_usage = {
                    "coding": {
                        "prompt_tokens": coding_prompt,
                        "completion_tokens": coding_completion,
                        "cost": coding_cost
                    },
                    "summarizer": {
                        "prompt_tokens": summarizer_prompt,
                        "completion_tokens": summarizer_completion,
                        "cost": summarizer_cost
                    },
                    "total": {
                        "prompt_tokens": coding_prompt + summarizer_prompt,
                        "completion_tokens": coding_completion + summarizer_completion,
                        "cost": coding_cost + summarizer_cost
                    }
                }

                action_log = {
                    "step": self.global_step,
                    "type": "code_execution",
                    "task": task,
                    "environment": environment,
                    "model": self.coding_model,
                    "save_path": coding_output_path,
                    "code": [],
                    "step_time": round(total_coding_time, 2),
                    "token_usage": token_usage
                }
                self.action_logs.append(action_log)

        except Exception as e:
            # Calculate time even for error case
            coding_agent_end_time = time.time()
            total_coding_time = coding_agent_end_time - coding_agent_start_time

            # Record error in action log even when failed
            error_msg = traceback.format_exc()

            # Increment global step counter for error case
            self.global_step += 1

            # Build empty token_usage for error case
            token_usage = {
                "coding": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                "summarizer": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                "total": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
            }

            action_log = {
                "step": self.global_step,
                "type": "code_execution",
                "task": task,
                "environment": environment,
                "model": self.coding_model,
                "save_path": coding_output_path,
                "code": [],  # Empty scripts on error, use "code" field to match langgraph_agent.py
                "step_time": round(total_coding_time, 2),
                "error": error_msg,
                "token_usage": token_usage
            }
            self.action_logs.append(action_log)
            summarized_history = f"ERROR: {str(e)}"

        finally:
            # Always update model-specific usage regardless of success or failure
            if coding_agent:
                coding_usage = coding_agent.get_total_usage()
                if self.coding_model in coding_usage:
                    if "coding" not in self.model_usage:
                        self.model_usage["coding"] = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
                    prompt_tokens = coding_usage[self.coding_model].get("prompt_tokens", 0)
                    completion_tokens = coding_usage[self.coding_model].get("completion_tokens", 0)
                    prompt_price, completion_price = get_price(self.coding_model)
                    cost = prompt_tokens * prompt_price + completion_tokens * completion_price
                    self.model_usage["coding"]["cost"] += cost
                    self.model_usage["coding"]["prompt_tokens"] += prompt_tokens
                    self.model_usage["coding"]["completion_tokens"] += completion_tokens
                    self.model_usage["coding"]["image_count"] += 1

            if summarizer:
                summarizer_usage = summarizer.get_total_usage()
                if self.summarizer_model in summarizer_usage:
                    if "summarizer" not in self.model_usage:
                        self.model_usage["summarizer"] = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
                    prompt_tokens = summarizer_usage[self.summarizer_model].get("prompt_tokens", 0)
                    completion_tokens = summarizer_usage[self.summarizer_model].get("completion_tokens", 0)
                    prompt_price, completion_price = get_price(self.summarizer_model)
                    cost = prompt_tokens * prompt_price + completion_tokens * completion_price
                    self.model_usage["summarizer"]["cost"] += cost
                    self.model_usage["summarizer"]["prompt_tokens"] += prompt_tokens
                    self.model_usage["summarizer"]["completion_tokens"] += completion_tokens

        screenshot = self.env.controller.get_screenshot()
        return f"# Response from coding agent: {summarized_history}"
    
    def _cua_to_pyautogui(self, action) -> str:
        """Convert an Action (dict **or** Pydantic model) into a pyautogui call."""
        def fld(key: str, default: Any = None) -> Any:
            return action.get(key, default) if isinstance(action, dict) else getattr(action, key, default)

        act_type = fld("type")
        if not isinstance(act_type, str):
            act_type = str(act_type).split(".")[-1]
        act_type = act_type.lower()

        if act_type in ["click", "double_click"]:
            button = fld('button', 'left')
            if button == 1 or button == 'left':
                button = 'left'
            elif button == 2 or button == 'middle':
                button = 'middle'
            elif button == 3 or button == 'right':
                button = 'right'

            if act_type == "click":
                return f"pyautogui.click({fld('x')}, {fld('y')}, button='{button}')"
            if act_type == "double_click":
                return f"pyautogui.doubleClick({fld('x')}, {fld('y')}, button='{button}')"
            
        if act_type == "scroll":
            cmd = ""
            if fld('scroll_y', 0) != 0:
                cmd += f"pyautogui.scroll({-fld('scroll_y', 0) / 100}, x={fld('x', 0)}, y={fld('y', 0)});"
            return cmd
        if act_type == "drag":
            path = fld('path', [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
            cmd = f"pyautogui.moveTo({path[0]['x']}, {path[0]['y']}, _pause=False); "
            cmd += f"pyautogui.dragTo({path[1]['x']}, {path[1]['y']}, duration=0.5, button='left')"
            return cmd

        if act_type == 'move':
            return f"pyautogui.moveTo({fld('x')}, {fld('y')})"

        if act_type == "keypress":
            keys = fld("keys", []) or [fld("key")]
            if len(keys) == 1:
                return f"pyautogui.press('{keys[0].lower()}')"
            else:
                return "pyautogui.hotkey('{}')".format("', '".join(keys)).lower()
            
        if act_type == "type":
            text = str(fld("text", ""))
            return "pyautogui.typewrite({:})".format(repr(text))
        
        if act_type == "wait":
            return "WAIT"
        
        return f"# Unknown action type: {act_type}"
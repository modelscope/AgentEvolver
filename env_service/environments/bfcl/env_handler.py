import json
import time
import os
from typing import Dict, List, Any, Optional, Union
import warnings
import tempfile
from pathlib import Path
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import _func_doc_language_specific_pre_processing

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
)
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.eval_checker.eval_runner import (
    relevance_file_runner,
    multi_turn_runner,
    get_handler,
    ast_file_runner,
)
from bfcl_eval.eval_checker.eval_runner_helper import record_result, record_cost_latency
from bfcl_eval.utils import (
    is_multi_turn,
    is_relevance_or_irrelevance,
    find_file_by_category,
    load_file,
)



# monkey patch to locate a possible answer path.
# users are expected to set this path manually in EnvHandler.
POSSIBLE_ANSWER_PATH = Path(
    os.path.join(__file__, "..", "..", "..", "..", "data", "possible_answer")
).resolve()


class EnvHandler:
    """
    A stateless standardized interface for bfcl v3 environment.
    Interacts with environment using chat messages format.
    This interface provides responses to assistant messages.
    """

    def __init__(
        self, model_name: str = "env_handler", answer_path: Path = POSSIBLE_ANSWER_PATH
    ):
        """
        Initialize the environment handler.

        Args:
            model_name: Name of the model to use. Defaults to "env_handler".
        """
        self.original_model_name = model_name
        self.model_name = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self._answer_path = answer_path
        if not self._answer_path.exists():
            raise ValueError(
                f"Answer path {self._answer_path} does not exist. Please refer to README.md for more information."
            )

    def interact(
        self, messages: List[Dict[str, Any]], test_entry: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Process one step in the conversation.
        Both single turn and multi turn are supported.

        Args:
            messages: List of conversation messages, with the last one being assistant response
            test_entry: Test entry containing initial_config, involved_classes, question etc.
            **kwargs: Additional arguments for compatibility

        Returns:
            Dict containing next message and tools if applicable
        """
        try:
            current_turn = self._get_current_turn(messages, test_entry)

            if not messages:
                return self._handle_user_turn(test_entry, current_turn)

            if messages[-1]["role"] != "assistant":
                return self._create_error_response(
                    "Last message must be from assistant"
                )

            if "tool_calls" in messages[-1] and len(messages[-1]["tool_calls"]) > 0:
                try:
                    tool_calls = messages[-1]["tool_calls"]
                    decoded_calls = self._convert_tool_calls_to_execution_format(
                        tool_calls
                    )
                    print(f"decoded_calls: {decoded_calls}")
                    # todo 实现decode_execute，返回prm
                    # if self.decode_execute(decoded_calls):
                    if is_empty_execute_response(decoded_calls):
                        warnings.warn(
                            f"is_empty_execute_response: {is_empty_execute_response(decoded_calls)}"
                        )
                        return self._handle_user_turn(test_entry, current_turn)
                    return self._handle_tool_calls(
                        tool_calls, decoded_calls, test_entry, current_turn
                    )
                except Exception as e:
                    warnings.warn(f"处理工具调用时发生错误: {str(e)}")
                    return self._handle_user_turn(test_entry, current_turn)
            else:
                return self._handle_user_turn(test_entry, current_turn)

        except Exception as e:
            return self._create_error_response(f"处理请求时发生错误: {str(e)}")

    def _get_current_turn(
        self, messages: List[Dict[str, Any]], test_entry: Dict[str, Any]
    ) -> int:
        """
        Get the current turn number in the conversation.

        Args:
            messages: List of conversation messages
            test_entry: Test entry containing conversation data

        Returns:
            Current turn number based on user messages count
        """
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        return len(user_messages)

    def _handle_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        decoded_calls: list[str],
        test_entry: Dict[str, Any],
        current_turn: int,
    ) -> Dict[str, Any]:
        """
        Handle tool calls from assistant.

        Args:
            tool_calls: List of tool calls in OpenAI format
            decoded_calls: List of decoded function calls
            test_entry: Test entry containing environment data
            current_turn: Current turn number

        Returns:
            Response containing tool execution results
        """
        execution_results, _ = execute_multi_turn_func_call(
            func_call_list=decoded_calls,
            initial_config=test_entry["initial_config"],
            involved_classes=test_entry["involved_classes"],
            model_name=self.model_name,
            test_entry_id=test_entry["id"],
            long_context=(
                "long_context" in test_entry["id"] or "composite" in test_entry["id"]
            ),
            is_evaL_run=False,
        )

        return self._create_tool_response(tool_calls, execution_results)

    def _handle_user_turn(
        self, test_entry: Dict[str, Any], current_turn: int
    ) -> Dict[str, Any]:
        """
        Handle user turn by returning appropriate content from test_entry["question"].
        For non-first turns, processes user query and tools.

        Args:
            test_entry: Test entry containing conversation data
            current_turn: Current turn number

        Returns:
            Response containing next user message and tools
        """
        try:
            current_turn_message = []
            tools = self._compile_tools(test_entry)
            questions = test_entry.get("question", [])
            holdout_function = test_entry.get("holdout_function", {})

            if str(current_turn) in holdout_function:
                test_entry["function"].extend(holdout_function[str(current_turn)])
                tools = self._compile_tools(test_entry)
                assert (
                    len(questions[current_turn]) == 0
                ), "Holdout turn should not have user message."
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]
                return self._create_user_response(current_turn_message, tools)
            if current_turn >= len(questions):
                return self._create_completion_response()

            current_turn_message = questions[current_turn]

            return self._create_user_response(current_turn_message, tools)

        except Exception as e:
            return self._create_error_response(f"处理用户轮次时发生错误: {str(e)}")

    def _compile_tools(self, test_entry: dict) -> list:
        """
        Compile functions into tools format.

        Args:
            test_entry: Test entry containing functions

        Returns:
            List of tools in OpenAI format
        """
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = _func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        return tools

    def _convert_tool_calls_to_execution_format(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Convert OpenAI format tool calls to execution format.

        Args:
            tool_calls: List of tool calls in OpenAI format

        Returns:
            List of function calls in string format
        """
        execution_list = []

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            function_name = function.get("name", "")

            try:
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    args_dict = json.loads(arguments)
                else:
                    args_dict = arguments

                args_str = ", ".join([f"{k}={repr(v)}" for k, v in args_dict.items()])
                execution_list.append(f"{function_name}({args_str})")

            except Exception as e:
                execution_list.append(f"{function_name}()")

        return execution_list

    def _create_tool_response(
        self, tool_calls: List[Dict[str, Any]], execution_results: List[str]
    ) -> Dict[str, Any]:
        """
        Create response for tool calls.

        Args:
            tool_calls: List of tool calls
            execution_results: List of execution results

        Returns:
            Response containing tool execution results
        """
        tool_messages = []
        for i, (tool_call, result) in enumerate(zip(tool_calls, execution_results)):
            tool_messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.get("id", f"call_{i}"),
                }
            )

        return {"messages": tool_messages}

    def _create_user_response(
        self, question_turn: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create response containing user message.

        Args:
            question_turn: List of messages for current turn
            tools: List of available tools

        Returns:
            Response containing user message and tools
        """
        user_content = ""
        for msg in question_turn:
            if msg["role"] == "user":
                user_content = msg["content"]
                break

        return {"messages": [{"role": "user", "content": user_content}], "tools": tools}

    def _create_completion_response(self) -> Dict[str, Any]:
        """
        Create response indicating conversation completion.

        Returns:
            Response with completion message
        """
        return {"messages": [{"role": "env", "content": "[CONVERSATION_COMPLETED]"}]}

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create response for error conditions.

        Args:
            error_message: Error message to include

        Returns:
            Response containing error message
        """
        return {"messages": [{"role": "env", "content": f"[ERROR] {error_message}"}]}

    def decode_execute(self, result):
        """
        Decode execute results for compatibility with evaluation framework.

        Args:
            result: Result to decode

        Returns:
            List of decoded function calls
        """
        return default_decode_execute_prompting(result)

    def evaluate(self, test_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate function for single test case.

        Args:
            test_entry: Test entry containing conversation_result and original_test_entry
                Expected format:
                {
                    "test_id": str,
                    "messages": List[Dict],
                    "turn_count": int,
                    "total_input_tokens": int,
                    "total_output_tokens": int,
                    "completed": bool,
                    "original_test_entry": Dict
                }
                or directly the conversation_result dict

        Returns:
            Evaluation results in format compatible with evaluate_task
        """
        try:
            conversation_result = test_entry
            original_test_entry = conversation_result.get("original_test_entry", {})

            if not conversation_result or not original_test_entry:
                return self._create_eval_error_result(
                    "Missing conversation_result or original_test_entry",
                    test_entry.get("test_id", "unknown"),
                )

            test_id = conversation_result.get("test_id", "unknown")
            messages = conversation_result.get("messages", [])

            category = test_id.rsplit("_", 1)[0] if "_" in test_id else test_id

            model_name = self.model_name
            # from bfcl_eval.model_handler.api_inference.qwq import QwenAPIHandler
            from bfcl_eval.model_handler.api_inference.qwen import QwenAPIHandler #### qwq->qwen
            
            # FIXME: missing parameter is_fc_model and registry_name. I am not sure what they are for, so I just set them to False and original_model_name.
            handler = QwenAPIHandler(
                self.model_name, temperature=1.0, is_fc_model=False, registry_name=self.original_model_name
            )  # FIXME: magic number

            model_result_data = self._convert_conversation_to_eval_format(
                conversation_result, original_test_entry
            )

            prompt_data = [original_test_entry]

            state = {"leaderboard_table": {}}
            record_cost_latency(
                state["leaderboard_table"], model_name, [model_result_data]
            )

            if is_relevance_or_irrelevance(category):
                accuracy, total_count = self._eval_relevance_test(
                    handler, model_result_data, prompt_data, model_name, category
                )
            else:
                # Find the corresponding possible answer file

                possible_answer_file = find_file_by_category(
                    category, self._answer_path
                )
                possible_answer = load_file(possible_answer_file, sort_by_id=True)
                possible_answer = [
                    item for item in possible_answer if item["id"] == test_id
                ]
                if is_multi_turn(category):
                    accuracy, total_count = self._eval_multi_turn_test(
                        handler,
                        model_result_data,
                        prompt_data,
                        possible_answer,
                        model_name,
                        category,
                    )
                else:
                    accuracy, total_count = self._eval_single_turn_test(
                        handler,
                        model_result_data,
                        prompt_data,
                        possible_answer,
                        model_name,
                        category,
                    )
            # print(f"model_result_data: {model_result_data}")
            # print(f"possible_answer: {possible_answer}") if possible_answer else None

            result = {
                "valid": True,
                "accuracy": accuracy,
                "total_count": total_count,
                "correct_count": int(accuracy * total_count),
                "test_category": category,
                "test_id": test_id,
                "model_name": model_name,
                "input_tokens": conversation_result.get("total_input_tokens", 0),
                "output_tokens": conversation_result.get("total_output_tokens", 0),
                "turn_count": conversation_result.get("turn_count", 0),
                "completed": conversation_result.get("completed", False),
            }

            return result

        except Exception as e:
            import traceback

            traceback.print_exc()
            return self._create_eval_error_result(
                f"Evaluation failed: {str(e)}",
                test_entry.get(
                    "test_id",
                    test_entry.get("conversation_result", {}).get("test_id", "unknown"),
                ),
            )

    def _create_eval_error_result(
        self, error_message: str, test_id: str
    ) -> Dict[str, Any]:
        """
        Create standardized error result for evaluation.

        Args:
            error_message: Error message to include
            test_id: ID of the test case

        Returns:
            Dictionary containing error result information
        """
        return {
            "valid": False,
            "error": error_message,
            "accuracy": 0.0,
            "total_count": 1,
            "correct_count": 0,
            "test_id": test_id,
            "model_name": self.model_name,
        }

    def _eval_relevance_test(
        self, handler, model_result_data, prompt_data, model_name, test_category
    ):
        """
        Evaluate relevance/irrelevance test.

        Args:
            handler: Model handler instance
            model_result_data: Model result data
            prompt_data: Prompt data
            model_name: Name of the model
            test_category: Category of the test

        Returns:
            Tuple of (accuracy, total_count)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir)
            accuracy, total_count = relevance_file_runner(
                handler=handler,
                model_result=[model_result_data],
                prompt=prompt_data,
                model_name=model_name,
                test_category=test_category,
                score_dir=score_dir,
            )
            self._capture_and_print_score_files(
                score_dir, model_name, test_category, "relevance"
            )
            return accuracy, total_count

    def _eval_multi_turn_test(
        self,
        handler,
        model_result_data,
        prompt_data,
        possible_answer,
        model_name,
        test_category,
    ):
        """
        Evaluate multi-turn test.

        Args:
            handler: Model handler instance
            model_result_data: Model result data
            prompt_data: Prompt data
            possible_answer: Possible answer data
            model_name: Name of the model
            test_category: Category of the test

        Returns:
            Tuple of (accuracy, total_count)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir)
            accuracy, total_count = multi_turn_runner(
                handler=handler,
                model_result=[model_result_data],
                prompt=prompt_data,
                possible_answer=possible_answer,
                model_name=model_name,
                test_category=test_category,
                score_dir=score_dir,
            )
            self._capture_and_print_score_files(
                score_dir, model_name, test_category, "multi_turn"
            )
            return accuracy, total_count

    def _eval_single_turn_test(
        self,
        handler,
        model_result_data,
        prompt_data,
        possible_answer,
        model_name,
        test_category,
    ):
        """
        Evaluate single-turn AST test.

        Args:
            handler: Model handler instance
            model_result_data: Model result data
            prompt_data: Prompt data
            possible_answer: Possible answer data
            model_name: Name of the model
            test_category: Category of the test

        Returns:
            Tuple of (accuracy, total_count)
        """
        language = "Python"
        if "java" in test_category.lower():
            language = "Java"
        elif "js" in test_category.lower() or "javascript" in test_category.lower():
            language = "JavaScript"

        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir)
            accuracy, total_count = ast_file_runner(
                handler=handler,
                model_result=[model_result_data],
                prompt=prompt_data,
                possible_answer=possible_answer,
                language=language,
                test_category=test_category,
                model_name=model_name,
                score_dir=score_dir,
            )
            self._capture_and_print_score_files(
                score_dir, model_name, test_category, "single_turn"
            )
            return accuracy, total_count

    def _capture_and_print_score_files(
        self, score_dir: Path, model_name: str, test_category: str, eval_type: str
    ):
        """
        Capture and print contents of score files written to score_dir.

        Args:
            score_dir: Directory containing score files
            model_name: Name of the model
            test_category: Category of the test
            eval_type: Type of evaluation (relevance/multi_turn/single_turn)
        """
        try:
            # print(f"\n=== {eval_type.upper()} Evaluation Result Files ===")
            # print(f"Model: {model_name}")
            # print(f"Test Category: {test_category}")
            # print(f"Evaluation Type: {eval_type}")

            for file_path in score_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(score_dir)
                    # print(f"\n--- File: {relative_path} ---")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if (
                            file_path.suffix == ".json"
                            or content.strip().startswith("{")
                            or content.strip().startswith("[")
                        ):
                            try:
                                import json

                                lines = content.strip().split("\n")
                                formatted_lines = []
                                for line in lines:
                                    if line.strip():
                                        parsed = json.loads(line)
                                        formatted_lines.append(
                                            json.dumps(
                                                parsed, ensure_ascii=False, indent=2
                                            )
                                        )
                                content = "\n".join(formatted_lines)
                            except json.JSONDecodeError:
                                pass

                        # print(content)

                    except UnicodeDecodeError:
                        print(f"[Binary file, size: {file_path.stat().st_size} bytes]")
                    except Exception as e:
                        print(f"[Error reading file: {str(e)}]")

            # print(f"=== {eval_type.upper()} Evaluation Result Files End ===\n")

        except Exception as e:
            print(f"Error capturing evaluation result files: {str(e)}")

    def _convert_conversation_to_eval_format(
        self, conversation_result: Dict[str, Any], original_test_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert conversation history to evaluation format.

        Args:
            conversation_result: Result from run_conversation
            original_test_entry: Original test entry data

        Returns:
            Data in format expected by multi_turn_runner or other runners
        """
        test_id = conversation_result.get("test_id", "unknown")
        messages = conversation_result.get("messages", [])

        test_category = test_id.rsplit("_", 1)[0] if "_" in test_id else test_id

        if is_multi_turn(test_category):
            turns_data = self._extract_multi_turn_responses(messages)
        else:
            turns_data = self._extract_single_turn_response(messages)

        model_result_data = {
            "id": test_id,
            "result": turns_data,
            "latency": conversation_result.get("total_latency", 0),
            "input_token_count": conversation_result.get("total_input_tokens", 0),
            "output_token_count": conversation_result.get("total_output_tokens", 0),
        }

        return model_result_data

    def _extract_multi_turn_responses(
        self, messages: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """
        Extract multi-turn responses from conversation messages.

        Args:
            messages: List of conversation messages

        Returns:
            List of turns, each turn is a list of function call strings
        """
        turns_data = []
        current_turn_responses = []

        i = 0
        while i < len(messages):
            message = messages[i]

            if message["role"] == "user":
                if current_turn_responses:
                    turns_data.append(current_turn_responses)
                    current_turn_responses = []

                i += 1
                while i < len(messages) and messages[i]["role"] == "assistant":
                    assistant_msg = messages[i]

                    if "tool_calls" in assistant_msg and assistant_msg["tool_calls"]:
                        for tool_call in assistant_msg["tool_calls"]:
                            formatted_call = self._format_single_tool_call_for_eval(
                                tool_call
                            )
                            if formatted_call:
                                current_turn_responses.append(formatted_call)

                    i += 1

                    while i < len(messages) and messages[i]["role"] == "tool":
                        i += 1
            else:
                i += 1

        if current_turn_responses:
            turns_data.append(current_turn_responses)

        return turns_data

    def _extract_single_turn_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract single-turn response from conversation messages.

        Args:
            messages: List of conversation messages

        Returns:
            String representation of the response
        """
        for message in reversed(messages):
            if message["role"] == "assistant":
                if "tool_calls" in message and message["tool_calls"]:
                    formatted_calls = []
                    for tool_call in message["tool_calls"]:
                        formatted_call = self._format_single_tool_call_for_eval(
                            tool_call
                        )
                        if formatted_call:
                            formatted_calls.append(formatted_call)
                    return "\n".join(formatted_calls) if formatted_calls else ""
                elif message.get("content"):
                    return message["content"]

        return ""

    def _format_single_tool_call_for_eval(self, tool_call: Dict[str, Any]) -> str:
        """
        Format a single tool call into string representation for evaluation.

        Args:
            tool_call: Single tool call in OpenAI format

        Returns:
            Formatted string representation
        """
        function = tool_call.get("function", {})
        function_name = function.get("name", "")

        try:
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                args_dict = json.loads(arguments)
            else:
                args_dict = arguments

            args_str = ", ".join([f"{k}={repr(v)}" for k, v in args_dict.items()])
            return f"{function_name}({args_str})"

        except Exception as e:
            return f"{function_name}()"


def env_step(
    messages: List[Dict[str, Any]],
    test_entry: Dict[str, Any],
    model: str = "env-handler",
    **kwargs,
) -> Dict[str, Any]:
    """
    Simplified interface for environment chat completion.

    Args:
        messages: List of conversation messages
        test_entry: Test entry containing conversation data
        model: Model name
        **kwargs: Additional arguments

    Returns:
        Response from environment handler
    """
    handler = EnvHandler(model)
    return handler.interact(messages, test_entry, **kwargs)

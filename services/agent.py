from typing import Any, Dict, List
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from services.additional_prompt import system_message
import time
from services.tools import tools, plan_steps
import json
import re
from services.llm_config import get_llm

llm = get_llm()

memory = MemorySaver()

# For persistence across runs:
# memory = SqliteSaver.from_conn_string("langgraph_agent.db")

agent_executor = create_react_agent(
    llm, tools, prompt=system_message, checkpointer=memory)


def extract_json_from_output(text: str) -> str:
    if not text:
        return None

    if "FINAL_PLAN_JSON:" in text:
        parts = text.split("FINAL_PLAN_JSON:", 1)
        if len(parts) > 1:
            json_text = parts[1].strip()

            code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
            code_block_match = re.search(code_block_pattern, json_text)
            if code_block_match:
                return code_block_match.group(1).strip()

            return json_text

    json_pattern = r"(\[[\s\S]*\]|\{[\s\S]*\})"
    json_matches = re.findall(json_pattern, text)
    if json_matches:
        return max(json_matches, key=len)

    return None


def agent(question: str, thread_id: str = "user_thread_1") -> Dict[str, Any]:
    global plan_steps
    plan_steps = []

    initial_state = {"messages": [("user", question)]}
    final_answer_content = None
    error_message = None

    print(
        f"\n========= Running Agent for Request (Thread: {thread_id}) =========\n")
    print(f"User Request: {question}")
    start_time = time.time()

    try:
        events = agent_executor.stream(
            initial_state,
            config={"configurable": {"thread_id": thread_id},
                    "recursion_limit": 50},
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()
            # Check the final message from the agent (could be in various formats)
            if isinstance(event["messages"][-1].content, str):
                content = event["messages"][-1].content
                # Check for the FINAL_PLAN_JSON marker or JSON pattern
                if "FINAL_PLAN_JSON" in content or content.strip().startswith("{") or content.strip().startswith("["):
                    final_answer_content = content

    except Exception as e:
        print(f"\n!!! Agent Execution Error: {e} !!!\n")
        import traceback
        traceback.print_exc()
        error_message = f"Agent execution failed: {e}"

    execution_time = time.time() - start_time
    print(
        f"\n========= Agent Execution Finished ({execution_time:.2f}s) =========\n")

    plan_json = None
    status = "Error"
    if final_answer_content:
        try:
            # Extract JSON from the content using our pattern matcher
            json_string = extract_json_from_output(final_answer_content)
            if not json_string:
                raise ValueError("Could not extract JSON from agent output")

            # Parse the extracted JSON
            plan_json = json.loads(json_string)

            # If this is a wrapper object with a 'plan' key, use that directly
            if isinstance(plan_json, dict) and "plan" in plan_json:
                status = "Plan Generated Successfully"
                print("Final Plan JSON:")
                print(json.dumps(plan_json, indent=2))
            # If it's a list, assume it's a list of steps and wrap in a 'plan' object
            elif isinstance(plan_json, list):
                plan_json = {"plan": plan_json}
                status = "Plan Generated Successfully"
                print("Final Plan JSON:")
                print(json.dumps(plan_json, indent=2))
            else:
                status = "Error: Invalid plan structure"
                error_message = "Agent produced invalid plan structure"

        except (json.JSONDecodeError, ValueError) as json_err:
            print(
                f"Error: Could not parse final JSON from agent output: {json_err}")
            print(f"Raw final content was:\n{final_answer_content}")
            # Try one more time with even more aggressive pattern matching
            try:
                # Print the full output for debugging
                print("Trying fallback JSON extraction...")
                # Find anything that looks like a JSON object or array
                json_pattern = r'(\{[^{]*?\}|\[[^[]*?\])'
                matches = re.findall(json_pattern, final_answer_content)
                if matches:
                    largest_match = max(matches, key=len)
                    print(f"Found potential JSON: {largest_match[:100]}...")
                    plan_json = json.loads(largest_match)
                    if isinstance(plan_json, dict) and "plan" in plan_json:
                        status = "Plan Generated Successfully (Fallback)"
                    elif isinstance(plan_json, list):
                        plan_json = {"plan": plan_json}
                        status = "Plan Generated Successfully (Fallback)"
                    else:
                        status = "Error: Invalid plan structure from fallback"
                        error_message = "Agent produced invalid plan structure"
                else:
                    error_message = f"Agent finished but failed to produce valid JSON after fallback: {json_err}"
                    status = "Error: Invalid JSON Output"
            except Exception as fallback_error:
                error_message = f"Agent finished but failed to produce valid JSON: {json_err}"
                status = "Error: Invalid JSON Output"
        except Exception as e:
            print(f"Unexpected error processing final output: {e}")
            error_message = f"Unexpected error processing final output: {e}"
            status = "Error: Processing Failed"

    elif error_message:
        status = error_message
    else:
        status = "Error: Agent finished without generating a final plan."
        print(status)

    return {
        "status": status,
        "plan": plan_json,
        "time_taken": f"{execution_time:.2f} seconds",
        "thread_id": thread_id
    }


if __name__ == "__main__":
    # Example Usage 1: YouTube Search
    # request1 = "Go to youtube, search for 'Langchain tutorials', wait a bit, then click the first video link."
    # result1 = agent(request1, thread_id="youtube_search_1")
    # print("\n--- Result 1 ---")
    # print(result1)

    # Example Usage 2: Veritasium Request
    # request2 = "Play Veritasium's latest video and like it."
    # result2 = agent(request2, thread_id="veritasium_like_1")
    # print("\n--- Result 2 ---")
    # # Pretty print the JSON if available
    # if result2.get("plan"):
    #     print("Status:", result2["status"])
    #     print("Plan:")
    #     print(json.dumps(result2["plan"], indent=2))
    #     print("Time Taken:", result2["time_taken"])
    # else:
    #     print(result2)

    # Example Usage 3: Simple Google Search
    request3 = "Open google and search for 'best python libraries 2024'."
    result3 = agent(request3, thread_id="google_search_1")
    if result3.get("plan"):
        print("Status:", result3["status"])
        print("Plan:")
        print(json.dumps(result3["plan"], indent=2))
        print("Time Taken:", result3["time_taken"])
    print("\n--- Result 3 ---")
    print(result3)

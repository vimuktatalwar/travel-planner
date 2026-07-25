import json
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from groq import Groq

# Configure the client (loads GROQ_API_KEY from .env)
load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


PLANNER_SYSTEM_PROMPT = """
You are a travel planning task-decomposition agent.

Convert the user's travel request into a complete JSON array of small,
actionable tasks.

Every task MUST contain exactly these fields:

{
  "task_id": integer,
  "task": string,
  "category": string,
  "city": string or null,
  "status": "pending",
  "depends_on": array of integer task IDs
}

Use sequential task IDs beginning with 1.

Allowed categories:
- research
- itinerary
- attraction
- transportation
- budget
- lodging
- food
- validation

Capture every city, duration, attraction requirement, transportation need,
and budget constraint.

Do not create the final itinerary.

Return only the valid JSON array. Do not include Markdown, explanations,
or code fences.
"""

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the  ChatCompletion API with the specified model and prompts.

    Args:
        system_prompt (str): System instruction prompt.
        user_prompt (str): User input prompt.

    Returns:
        str: The response content from the model.
    """
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1500,
        n=1,
    )
    completion_text = response.choices[0].message.content
    return completion_text

def generate_travel_tasks(user_request: str) -> List[Dict[str, Any]]:
    if not user_request.strip():
        raise ValueError("User request cannot be empty.")

    json_response = call_llm(PLANNER_SYSTEM_PROMPT, user_request)

    try:
        tasks = json.loads(json_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON response from LLM:\n{json_response}"
        ) from e

    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task list is empty or not a list.")

    allowed_categories = {
        "research",
        "itinerary",
        "attraction",
        "transportation",
        "budget",
        "lodging",
        "food",
        "validation",
    }

    normalized_tasks = []
    used_task_ids = set()

    for index, original_task in enumerate(tasks, start=1):
        if not isinstance(original_task, dict):
            raise ValueError(
                f"Task is not a dictionary: {original_task}"
            )

        task = original_task.copy()

        # Normalize alternate category key.
        if "type" in task and "category" not in task:
            task["category"] = task.pop("type")

        # Add task_id when the LLM omits it.
        task_id = task.get("task_id")

        if not isinstance(task_id, int) or task_id in used_task_ids:
            task_id = index

            while task_id in used_task_ids:
                task_id += 1

            task["task_id"] = task_id

        used_task_ids.add(task_id)

        # Add defaults for optional fields.
        task.setdefault("city", None)
        task.setdefault("status", "pending")
        task.setdefault("depends_on", [])

        # Validate required content.
        if not isinstance(task.get("task"), str) or not task["task"].strip():
            raise ValueError(
                f"Task is missing a valid task description: {task}"
            )

        if not isinstance(task.get("category"), str):
            raise ValueError(
                f"Task is missing a valid category: {task}"
            )

        category = task["category"].lower().strip()
        task["category"] = category

        if category not in allowed_categories:
            raise ValueError(
                f"Unsupported category '{category}': {task}"
            )

        if not isinstance(task["depends_on"], list):
            raise ValueError(
                f"'depends_on' must be a list: {task}"
            )

        normalized_tasks.append(task)

    return normalized_tasks

def execute_task(task: dict) -> str:
    """
    Simulate execution of a single travel planning task by sending it to an LLM executor prompt.
    This mock function is a placeholder for actual LLM interaction.

    Args:
        task (dict): A single task dictionary.

    Returns:
        str: Simulated execution result or response.
    """
    # Build a specialized prompt based on task details (mocked)
    system_prompt = (
        "You are a travel planning task executor. Execute the given small actionable task. "
        "Provide the best answer or data required without creating the full itinerary."
    )
    user_prompt = f"Execute this task: {task['task']}"

    # Mocked response simulating the execution result
    return f"Executed: {task['task']}"


def check_task_execution(task: dict) -> bool:
    """
    Simulate verifying if task execution succeeded by sending its result to an LLM checker prompt.
    This mock function determines success or failure based on keywords in the result string.

    Args:
        task (dict): A single task dictionary, expected to contain a 'result' field.

    Returns:
        bool: True if execution succeeded, False otherwise.
    """
    failure_indicators = [
        "no tickets available",
        "failed",
        "error",
        "not found",
        "unavailable",
        "unable to",
        "cannot",
    ]

    # Extract result text and normalize it
    result_text = task.get('result', '').lower()

    # If any failure indicator appears in the result, flag as failure
    for indicator in failure_indicators:
        if indicator in result_text:
            return False

    # Otherwise, consider success
    return True


def replan_tasks_on_failure(tasks: list[dict], failed_task: dict) -> list[dict]:
    """
    Simulate dynamically rewriting the remaining checklist items when a task execution fails.

    Args:
        tasks (list[dict]): Original list of tasks as dictionaries.
        failed_task (dict): The task dictionary that failed.

    Returns:
        list[dict]: Updated task list after replanning.
    """
    # For demonstration, just add a new task indicating replanning needed due to failure
    replan_notice_task = {
        "task_id": max(task["task_id"] for task in tasks) + 1,
        "task": f"Replan remaining tasks due to failure in task '{failed_task['task']}'",
        "category": "validation",
        "city": None,
        "status": "pending",
        "depends_on": [],
    }
    # Append replanning notice task at the end
    return tasks + [replan_notice_task]


def execute_and_check_tasks(tasks: list[dict]) -> list[dict]:
    """
    Execute tasks, check each execution, and dynamically replan if failures are detected.

    Args:
        tasks (list[dict]): List of travel planning tasks.

    Returns:
        list[dict]: List of tasks with execution results, possibly appended with replanning instructions.
    """
    executed_tasks = []

    for task in tasks:
        # Execute the task
        try:
            result = execute_task(task)
        except Exception as e:
            result = f"Execution failed: {str(e)}"

        # Append result to task copy
        task_with_result = task.copy()
        task_with_result['result'] = result
        executed_tasks.append(task_with_result)

        # Check if execution succeeded
        if not check_task_execution(task_with_result):
            # If failed, replan remaining tasks dynamically
            remaining_tasks = tasks[tasks.index(task)+1:]
            replanned = replan_tasks_on_failure(remaining_tasks, task_with_result)
            # Append replanned tasks (excluding original remaining to avoid duplication)
            executed_tasks.extend(replanned)
            break  # Stop executing further original tasks after failure

    return executed_tasks


# Demonstration using the example user request.
if __name__ == "__main__":
    user_request = (
        "I want to go to Dallas for 2 days, L.A for 2, and Orlando for 4, "
        "but I need to make sure I see a theme park in each and budget my flight tickets."
    )
    try:
        tasks = generate_travel_tasks(user_request)
        final_tasks = execute_and_check_tasks(tasks)
        print(json.dumps(final_tasks, indent=4))
    except ValueError as err:
        print(f"Error: {err}")
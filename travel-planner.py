import json
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

# Configure the client (loads GROQ_API_KEY from .env)
load_dotenv()
client_groq = Groq(api_key=os.environ["GROQ_API_KEY"])
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

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
  "depends_on": array of integer task IDs
}

Use sequential task IDs beginning with 1.

Allowed categories:
- attraction
- transportation
- lodging


Capture every city, duration, attraction requirement, and transportation need.
Do not book hotels, flights, or tickets. 
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
    """
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=1500,
        )
    except Exception as exc:
            raise RuntimeError(
            f"OpenAI API request failed: {exc}"
            ) from exc

    completion_text = response.output_text

    if not completion_text or not completion_text.strip():
        raise RuntimeError("OpenAI returned an empty response.")

    return completion_text.strip()

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
        "attraction",
        "transportation",
        "lodging",
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
    Execute one travel-planning task using the Groq-hosted Llama model.

    Args:
        task: A travel-planning task containing fields such as
            task_id, task, category, city, and depends_on.

    Returns:
        The LLM-generated result for the task.

    Raises:
        ValueError: If the task is invalid or the LLM returns no content.
        RuntimeError: If the Groq API request fails.
    """
    if not isinstance(task, dict):
        raise ValueError("Task must be a dictionary.")

    task_description = task.get("task")

    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("Task must contain a non-empty 'task' field.")

    system_prompt = """
You are a travel-planning task execution agent.

Execute one small travel-planning task at a time.

Rules:
1. Answer only the specific task provided.
2. Do not create the complete trip itinerary.
3. Give practical and concise results.
4. Include estimated prices when the task requests budgeting.
5. Clearly label prices, schedules, and travel times as estimates when they
   have not been verified with live data.
6. Do not claim that reservations or tickets are available unless live
   availability has been provided.
7. When the task concerns an attraction, provide its name, city, why it
   satisfies the requirement, and useful planning notes.
8. When the task concerns transportation, provide the route, transportation
   type, approximate duration, and estimated cost.
9. When the task concerns budgeting, show the calculation clearly.
10. If the task cannot be completed with the available information, explain
    exactly what additional information is needed.

Return plain text without Markdown code fences.
""".strip()

    task_payload = {
        "task_id": task.get("task_id"),
        "task": task_description,
        "category": task.get("category"),
        "city": task.get("city"),
        "depends_on": task.get("depends_on", []),
    }

    user_prompt = (
        "Execute the following travel-planning task:\n\n"
        f"{json.dumps(task_payload, indent=2, ensure_ascii=False)}"
    )

    try:
        result = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        raise RuntimeError(
            f"LLM execution failed for task {task.get('task_id')}: {exc}"
        ) from exc

    if not result or not result.strip():
        raise ValueError(
            f"The LLM returned an empty result for task {task.get('task_id')}."
        )

    return result.strip()


def check_task_execution(task: dict) -> bool:
    """
    Use the Llama model to determine whether a travel task was
    completed successfully.

    Args:
        task: A task dictionary containing the original task and
            its execution result.

    Returns:
        True when the checker determines that the task succeeded.
        False when the task failed, is incomplete, or cannot be verified.

    Raises:
        ValueError: If the task or result is invalid.
        RuntimeError: If the LLM checker request fails or returns invalid JSON.
    """
    if not isinstance(task, dict):
        raise ValueError("Task must be a dictionary.")

    task_description = task.get("task")
    result = task.get("result")

    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("Task must contain a non-empty 'task' field.")

    if not isinstance(result, str) or not result.strip():
        return False

    system_prompt = """
You are a quality-control checker for a travel-planning agent.

Determine whether the execution result successfully completes the original
travel-planning task.

A task is successful only when:

1. The result directly addresses the requested task.
2. The result contains enough useful information to complete the task.
3. The result does not report an error, failure, unavailable information,
   or inability to complete the task.
4. The result does not merely repeat the task.
5. Required information such as a location, route, cost estimate, attraction,
   duration, or calculation is present when requested.
6. Estimates are clearly identified as estimates when live information was
   not verified.
7. The result does not claim live availability, exact pricing, or confirmed
   reservations without supporting data.

Return only valid JSON using exactly this format:

{
  "success": true,
  "reason": "Brief explanation"
}

Do not include Markdown, code fences, or additional text.
""".strip()

    checker_payload = {
        "task_id": task.get("task_id"),
        "task": task_description,
        "category": task.get("category"),
        "city": task.get("city"),
        "result": result,
    }

    user_prompt = (
        "Evaluate whether this travel-planning task was completed successfully:\n\n"
        f"{json.dumps(checker_payload, indent=2, ensure_ascii=False)}"
    )

    try:
        checker_response = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        raise RuntimeError(
            f"LLM checker failed for task {task.get('task_id')}: {exc}"
        ) from exc

    if not checker_response or not checker_response.strip():
        raise RuntimeError(
            f"LLM checker returned an empty response for task "
            f"{task.get('task_id')}."
        )

    try:
        checker_data = json.loads(checker_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM checker returned invalid JSON:\n{checker_response}"
        ) from exc

    success = checker_data.get("success")

    if not isinstance(success, bool):
        raise RuntimeError(
            "LLM checker response must contain a Boolean 'success' field."
        )

    # Store the checker explanation for debugging and later replanning.
    task["check_reason"] = checker_data.get(
        "reason",
        "No checker explanation was provided.",
    )

    return success

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
        task_succeeded = False
        try:
            task_succeeded = check_task_execution(task_with_result)
        except Exception as exc:
            task_succeeded = False
            task_with_result["check_reason"] = (
                f"Checker could not evaluate the task: {exc}"
        )

        if not task_succeeded:
            remaining_tasks = tasks[tasks.index(task) + 1:]
            replanned = replan_tasks_on_failure(
            remaining_tasks,
            task_with_result        )
            executed_tasks.extend(replanned)
            break
      

    return executed_tasks


# Demonstration using the example user request.
if __name__ == "__main__":
    user_request = (
        "I want to plan itinerary to go from my home in Boston to Dallas for 2 days, L.A for 2, and Orlando for 2, "
        "but I need to make sure I see most popular theme park in each city,"
        "Keep one day for theme park and one day for other attarctions in each city. Advice other attracions in each city,"
         "and Travel dates are from September 1 to September 6, 2 people will be traveling, and I want to stay in 3-star hotels and fly economy class." 
    )
    try:
        tasks = generate_travel_tasks(user_request)
        final_tasks = execute_and_check_tasks(tasks)
       # print(json.dumps(final_tasks, indent=4))

        prompt1 = f"Summarize the itinerary in tabular format, include date, city, task, category, attraction names from result: {final_tasks}"

        response1 = client_groq.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt1}]
        )
        summary = response1.choices[0].message.content.strip()
        print(f"Summary: {summary}")

    except ValueError as err:
        print(f"Error: {err}")
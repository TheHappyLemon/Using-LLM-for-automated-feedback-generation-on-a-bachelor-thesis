from src.code.functions import get_prompt, get_texts, get_topics, save_used_prompts_not_divided, make_prompt, prepare_prompts, save_used_prompts

def main():

    introduction_texts_divided = get_texts()
    topics = get_topics()

    REFINE_BEFORE_GOAL            = get_prompt("REFINE_BEFORE_GOAL")
    REFINE_GOAL_WITH_PRECEDING    = get_prompt("REFINE_GOAL_WITH_PRECEDING_TEXT")
    REFINE_GOAL_WITHOUT_PRECEDING = get_prompt("REFINE_GOAL_WITHOUT_PRECEDING_TEXT")
    REFINE_TASKS_WITH_GOAL        = get_prompt("REFINE_TASKS_WITH_GOAL")
    REFINE_TASKS_WITHOUT_GOAL     = get_prompt("REFINE_TASKS_WITHOUT_GOAL")
    REFINE_AFTER_TASKS            = get_prompt("REFINE_AFTER_TASKS")

    prompts_part_refinement = prepare_prompts(
        introduction_texts_divided, topics,
        REFINE_BEFORE_GOAL, REFINE_GOAL_WITH_PRECEDING,
        REFINE_GOAL_WITHOUT_PRECEDING, REFINE_TASKS_WITH_GOAL,
        REFINE_TASKS_WITHOUT_GOAL, REFINE_AFTER_TASKS
    )

    # these are used in few-shot
    del prompts_part_refinement[1]
    del prompts_part_refinement[5]
    del prompts_part_refinement[7]
    del prompts_part_refinement[43]

    amount = 0

    for p in prompts_part_refinement:

        before_goal = prompts_part_refinement[p].get("BeforeGoal")
        goal = prompts_part_refinement[p].get("Goal")
        tasks = prompts_part_refinement[p].get("Tasks")
        after_tasks = prompts_part_refinement[p].get("AfterTasks")

        if before_goal != "":
            amount = amount + 5
        if goal != "":
            amount = amount + 3
            if before_goal != "":
                amount = amount + 1
        if tasks != "":
            amount = amount + 6
            if goal != "":
                amount = amount + 1
        if after_tasks != "":
            amount = amount + 3
    print(f"In total: {amount} questioons")
                

# python -m src.code.generating.few-shot.calc_amount_of_questions_in_promts
if __name__ == "__main__":
    main()
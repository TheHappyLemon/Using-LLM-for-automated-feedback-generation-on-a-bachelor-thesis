BEFORE_GOAL = """[
    {
        "question": "Significance",
        "feedback": "",
        "complies": {SIGNIFICANCE_COMPLIES}
    },
    {
        "question": "State-of-the-art",
        "feedback": "",
        "complies": {STATEOFTHEART_COMPLIES}
    },
    {
        "question": "Gap",
        "feedback": "",
        "complies": {GAP_COMPLIES}
    },
    {
        "question": "Problem",
        "feedback": "",
        "complies": {PROBLEM_COMPLIES}
    },
    {
        "question": "References",
        "feedback": "",
        "complies": {REFERENCES_COMPLIES}
    }
]"""

AFTER_TASKS = """[
    {
        "question": "Chapters",
        "feedback": "",
        "complies": {CHAPTERS_COMPLIES}
    },
    {
        "question": "Description",
        "feedback": "",
        "complies": {DESCRIPTION_COMPLIES}
    },
    {
        "question": "Structure",
        "feedback": "",
        "complies": {STRUCTURE_AFTER_TASKS_COMPLIES}
    }
]"""

GOAL_WITH_PRECEDING = """[
    {
        "question": "Purpose",
        "feedback": "",
        "complies": {PURPOSE_COMPLIES}
    },
    {
        "question": "Intention",
        "feedback": "",
        "complies": {INTENTION_COMPLIES}
    },
    {
        "question": "Structure",
        "feedback": "",
        "complies": {STRUCTURE_GOAL_COMPLIES}
    },
    {
        "question": "Congruence",
        "feedback": "",
        "complies": {CONGRUENCE_COMPLIES}
    }
]"""

GOAL_WITHOUT_PRECEDING = """[
    {
        "question": "Purpose",
        "feedback": "",
        "complies": {PURPOSE_COMPLIES}
    },
    {
        "question": "Intention",
        "feedback": "",
        "complies": {INTENTION_COMPLIES}
    },
    {
        "question": "Structure",
        "feedback": "",
        "complies": {STRUCTURE_GOAL_COMPLIES}
    }
]"""

TASKS_WITH_GOAL = """[
    {
        "question": "Outlook",
        "feedback": "",
        "complies": "{OUTLOOK_COMPLIES}"
    },
    {
        "question": "Quantity",
        "feedback": "{QUANTITY}",
        "complies": "{QUANTITY_COMPLIES}"
    },
    {
        "question": "Completeness",
        "feedback": "",
        "complies": "{COMPLETENESS_COMPLIES}"
    },
    {
        "question": "Format",
        "feedback": "",
        "complies": "{FORMAT_COMPLIES}"
    },
    {
        "question": "Structure",
        "feedback": "",
        "complies": "{STRUCTURE_TASKS_COMPLIES}"
    },
    {
        "question": "Clarity",
        "feedback": "",
        "complies": "{CLARITY_COMPLIES}"
    },
    {
        "question": "Relevance",
        "feedback": "",
        "complies": "{RELEVANCE_COMPLIES}"
    }
]"""

TASKS_WITHOUT_GOAL = """[
    {
        "question": "Outlook",
        "feedback": "",
        "complies": "{OUTLOOK_COMPLIES}"
    },
    {
        "question": "",
        "feedback": "{QUANTITY}",
        "complies": "{QUANTITY_COMPLIES}"
    },
    {
        "question": "Completeness",
        "feedback": "",
        "complies": "{COMPLETENESS_COMPLIES}"
    },
    {
        "question": "Format",
        "feedback": "",
        "complies": "{FORMAT_COMPLIES}"
    },
    {
        "question": "Structure",
        "feedback": "",
        "complies": "{STRUCTURE_TASKS_COMPLIES}"
    },
    {
        "question": "Clarity",
        "feedback": "",
        "complies": "{CLARITY_COMPLIES}"
    }
]"""
import json_repair
from src.data.constants import BASE_PATH
import os
import json

def main():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "feedback": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                        {"type": "null"}
                    ],
                    "default": None
                },
                "complies": {
                    "type": "boolean",
                    "default": False
                }
            },
            "required": ["question"],
            "additionalProperties": False
        }
    }
    json_str = '''
```json
[
    {
        "question": "Outlook",
        "feedback": "The tasks are clearly organized into a numbered list.",
        "complies": true
    },
    {
        "question": "Quantity",
        "feedback": "4",
        "complies": false
    },
    {
        "question": "Completeness",
        "feedback": "The tasks cover research, analysis, and development, but they lack a clear validation step where the developed prototype is compared against existing solutions or benchmarks to demonstrate its effectiveness.",
        "complies": false
    },
    {
        "question": "Format",
        "feedback": "Each task is contained within a single sentence.",
        "complies": true
    },
    {
        "question": "Structure",
        "feedback": "The tasks consistently use infinitive verbs (Apkopot, Analizēt, Izstrādāt) at the beginning.",
        "complies": true
    },
    {
        "question": "Clarity",
        "feedback": "The language is precise, professional, and the sentences are straightforward.",
        "complies": true
    },
    {
        "question": "Relevance",
        "feedback": "The tasks follow a logical progression toward the goal, covering literature review, analysis of existing methods, and the development of the prototype. However, it lacks a final task for validating the results.",
        "complies": false
    }
]
```
'''
    decoded_object, logs = json_repair.loads(json_str, schema=schema, schema_repair_mode='salvage', logging=True)
    print(type(decoded_object), decoded_object)
    with open(os.path.join(BASE_PATH, "src", "code", "parsing", "test_result.json"), 'w', encoding='utf-8') as f:
        json.dump(decoded_object, f)
    for l in logs:
        print(l)

def kek():
    return [], True, False

if __name__ == "__main__":
    main()
    
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
        "question": "Purpose",
        "feedback": "The goal is clear and measurable, as the development of software can be verified through its functionality and completion.",
        "complies": true
    },
    {
        "question": "Intention",
        "feedback": "The goal focuses on the outcome (developing software) without getting bogged down in the specific technical implementation or tools used.",
        "complies": true
    },
    {
        "question": "Structure",
        "feedback": "The text is a single, concise, and direct sentence that clearly communicates the objective.",
        "complies": true
    },
    /// lalalalalal
    {
        "question": "Congruence"
        "feedback": "The goal perfectly aligns with the preceding text, which identifies the need for training in computer assembly and proposes AR technology as the solution.",
        "complies": true
    }
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
    #main()
    (obj, bool1, bool2) = kek()
    print(type(obj), obj, type(bool1), bool1, type(bool2), bool2)
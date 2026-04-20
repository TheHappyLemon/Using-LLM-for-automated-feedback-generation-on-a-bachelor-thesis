import demjson3
import json5

def main():

    def repair_json(s: str) -> str:
        # Remove obvious comment types (json5 can handle them, but safe anyway)
        
        # Balance brackets
        stack = []
        for char in s:
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack and ((char == "}" and stack[-1] == "{") or
                            (char == "]" and stack[-1] == "[")):
                    stack.pop()
                else:
                    # skip extra closing bracket
                    s = s.replace(char, '', 1)

        # Add missing closing brackets
        for open_bracket in reversed(stack):
            s += "}" if open_bracket == "{" else "]"

        return s


    def parse_loose_json(s: str):
        repaired = repair_json(s)
        return json5.loads(repaired)

if __name__ == "__main__":
    main()
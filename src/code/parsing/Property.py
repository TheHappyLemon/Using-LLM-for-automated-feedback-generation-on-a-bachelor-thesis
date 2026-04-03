class Property:
    def __init__(self, value : int = 0, feedback : str = "") -> None:
        self.value = value
        self.feedback = feedback
    
    def __str__(self) -> str:
        return f"'value : {self.value}; feedback : {self.feedback}'"
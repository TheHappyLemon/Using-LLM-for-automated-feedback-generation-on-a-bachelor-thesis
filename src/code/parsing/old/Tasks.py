from src.code.parsing.old.Part import *
from src.code.parsing.old.Property import Property

class Tasks(Part):

    def __init__(self, Outlook : Property | None = None, Quantity : Property | None = None, Completeness : Property | None = None, Format : Property | None = None, Structure : Property | None = None, Clarity : Property | None = None, Relevance : Property | None = None):
        
        if Outlook is None:
            Outlook = Property()
        if Quantity is None:
            Quantity = Property()
        if Completeness is None:
            Completeness = Property()
        if Format is None:
            Format = Property()
        if Structure is None:
            Structure = Property()
        if Clarity is None:
            Clarity = Property()
        if Relevance is None:
            Relevance = Property()
        
        self.Outlook = Outlook
        self.Quantity = Quantity
        self.Completeness = Completeness
        self.Format = Format
        self.Structure = Structure
        self.Clarity = Clarity
        self.Relevance = Relevance

        self.questions_mapping = {
            "Outlook" : "Is the text organized into clearly separated items in form of a NUMBERED list?",
            "Quantity" : "Count the number of individual tasks in the text. Output only the number in 'feedback' field of JSON. Text complies if it has 5, 6 or 7 tasks.",
            "Completeness" : "Does the text describe individual tasks that represent a complete research process, including theoretical analysis, practical research (experiments) or implementation (of solution), and validation of results — for example, by comparing findings with prior studies, existing solutions, or established benchmarks? By validation we don't mean just something like developed software testing. If some kind of solution is developed, validation means that it must be compared to existing solutions or, at the very minimum, it must be demonstrated that is actually solves a practical problem. If some existing solutions are compared, validation means that the results of the comparison must be contrasted with results from other authors.",
            "Format" : "Does the text describe individual tasks that are exactly one sentence long and do not span over several sentences?",
            "Structure" : "Does the text consistently use infinitive verbs (e.g., \"to analyze\", \"to develop\", \"to explore\") at the beginning of sentences or phrases?",
            "Clarity" : "Does the text consist of short, straightforward sentences that use concise and precise language?",
            "Relevance" : "Does the text present a set of actions or objectives in an order that demonstrably contributes to reaching the stated thesis goal? If the goal is to develop a solution, the tasks should include research of existing solutions and development of the new solution as well as its validation. If the goal is about evaluating or comparing existing solutions, the tasks should include preparations for experiments, doing the experiments, analysis of the obtained results as well as their validation."
        }

    def to_bool(self, quantity_already_bool : bool = False):
        # Because one human provided amount of tasks, but another already binary answer
        if quantity_already_bool:
            self.Quantity.value = bool(self.Quantity.value)
        else:
            self.Quantity.value = self.Quantity.value >= 5 and self.Quantity.value <= 7 # pyright: ignore[reportAttributeAccessIssue]
        super().to_bool()

    def load(self, json_arr : list[QuestionAnswer], **kwargs):
        # If text, to whom this tasks section begins, has no 'Goal' text, then answer should not contain all defined question
        has_goal = kwargs.get("has_goal", False)
        if has_goal:
            super().load(json_arr)
        else:
            super().load(json_arr, mandatory_questions=["Outlook", "Quantity", "Completeness", "Format", "Structure", "Clarity"])
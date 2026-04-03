from __future__ import annotations

from src.code.parsing.old.AfterTasks import AfterTasks
from src.code.parsing.old.BeforeGoal import BeforeGoal
from src.code.parsing.old.Goal import Goal
from src.code.parsing.old.Tasks import Tasks
from src.code.parsing.old.Property import Property

class EvaluationRow:
    def __init__(self, author : str):
        self.Nr = -1
        self.beforeGoal = BeforeGoal()
        self.goal = Goal()
        self.tasks = Tasks()
        self.afterTasks = AfterTasks()

    def load_1(
        self,
        Nr: int,
        # BeforeGoal
        Significance: int,
        State_of_the_art: int,
        Gap: int,
        Problem: int,
        References: int,
        # Goal
        Purpose: int,
        Intention: int,
        Structure_goal: int,
        Congruence: int,
        # Tasks
        Outlook: int,
        Quantity: int,
        Completeness: int,
        Format: int,
        Structure_tasks: int,
        Clarity: int,
        Relevance: int,
        # AfterTasks
        Chapters: int,
        Description: int,
        Structure_aftertasks: int
    ):
        self.Nr = Nr
        self.beforeGoal = BeforeGoal(Property(Significance), Property(State_of_the_art), Property(Gap), Property(Problem), Property(References))
        self.goal = Goal(Property(Purpose), Property(Intention), Property(Structure_goal), Property(Congruence))
        self.tasks = Tasks(Property(Outlook), Property(Quantity), Property(Completeness), Property(Format), Property(Structure_tasks), Property(Clarity), Property(Relevance))
        self.afterTasks = AfterTasks(Property(Chapters), Property(Description), Property(Structure_aftertasks))

    def load_2(
        self,
        Nr: int,
        beforeGoal: BeforeGoal | None = None,
        goal: Goal | None = None,
        tasks: Tasks | None = None,
        afterTasks: AfterTasks | None = None
    ):

        if beforeGoal is None:
            beforeGoal = BeforeGoal()
        if goal is None:
            goal = Goal()
        if tasks is None:
            tasks = Tasks()
        if afterTasks is None:
            afterTasks = AfterTasks()

        self.Nr = Nr
        self.beforeGoal = beforeGoal
        self.goal = goal
        self.tasks = tasks
        self.afterTasks = afterTasks

    def to_bool(self, quantity_already_bool : bool = False):
        self.beforeGoal.to_bool()
        self.goal.to_bool() 
        self.tasks.to_bool(quantity_already_bool)
        self.afterTasks.to_bool()

    def to_int(self):
        self.beforeGoal.to_int()
        self.goal.to_int() 
        self.tasks.to_int()
        self.afterTasks.to_int()

    def compare(self, other : EvaluationRow):
        pass

    def to_str_compact(self):
        pass

    def __str__(self) -> str:
        return (
            f"EvaluationRow Nr {self.Nr}:\n"
            f"  Before Goal:\n"
            f"    Significance: {self.beforeGoal.Significance.value}\n"
            f"    State-of-the-art: {self.beforeGoal.State_of_the_art.value}\n" # type: ignore
            f"    Gap: {self.beforeGoal.Gap.value}\n" # type: ignore
            f"    Problem: {self.beforeGoal.Problem.value}\n" # type: ignore
            f"    References: {self.beforeGoal.References.value}\n"  # type: ignore
            f"  Goal:\n"
            f"    Purpose: {self.goal.Purpose.value}\n"  # type: ignore
            f"    Intention: {self.goal.Intention.value}\n"  # type: ignore
            f"    Structure: {self.goal.Structure.value}\n"  # type: ignore
            f"    Congruence: {self.goal.Congruence.value}\n"  # type: ignore
            f"  Tasks:\n"
            f"    Outlook: {self.tasks.Outlook.value}\n"  # type: ignore
            f"    Quantity: {self.tasks.Quantity.value}\n"  # type: ignore
            f"    Completeness: {self.tasks.Completeness.value}\n"  # type: ignore
            f"    Format: {self.tasks.Format.value}\n"  # type: ignore
            f"    Structure: {self.tasks.Structure.value}\n"  # type: ignore
            f"    Clarity: {self.tasks.Clarity.value}\n"  # type: ignore
            f"    Relevance: {self.tasks.Relevance.value}\n"  # type: ignore
            f"  After Tasks:\n"
            f"    Chapters: {self.afterTasks.Chapters.value}\n"  # type: ignore
            f"    Description: {self.afterTasks.Description.value}\n"  # type: ignore
            f"    Structure: {self.afterTasks.Structure.value}\n"  # type: ignore
        )
from src.code.parsing.old.Part import *
from src.code.parsing.old.Property import Property

class Full(Part):

    def __init__(self,
            # BeforeGoal
            Significance : Property | None = None, State_of_the_art : Property | None = None, Gap : Property | None = None, Problem : Property | None = None, References : Property | None = None,
            # Goal
            Purpose : Property | None = None, Intention : Property | None = None, Structure_goal : Property | None = None, Congruence : Property | None = None,
            # Tasks
            Outlook : Property | None = None, Quantity : Property | None = None, Completeness : Property | None = None, Format : Property | None = None, Structure_tasks : Property | None = None, Clarity : Property | None = None, Relevance : Property | None = None,
            # AfterTasks
            Chapters : Property | None = None, Description : Property | None = None, Structure_aftertasks : Property | None = None
        ):
        
        # BeforeGoal
        if Significance is None:
            Significance = Property()
        if State_of_the_art is None:
            State_of_the_art = Property()
        if Gap is None:
            Gap = Property()
        if Problem is None:
            Problem = Property()
        if References is None:
            References = Property()
        # Goal
        if Purpose is None:
            Purpose = Property()
        if Intention is None:
            Intention = Property()
        if Structure_goal is None:
            Structure_goal = Property()
        if Congruence is None:
            Congruence = Property()
        # Tasks
        if Outlook is None:
            Outlook = Property()
        if Quantity is None:
            Quantity = Property()
        if Completeness is None:
            Completeness = Property()
        if Format is None:
            Format = Property()
        if Structure_tasks is None:
            Structure_tasks = Property()
        if Clarity is None:
            Clarity = Property()
        if Relevance is None:
            Relevance = Property()
        # AfterTasks
        if Chapters is None:
            Chapters = Property()
        if Description is None:
            Description = Property()
        if Structure_aftertasks is None:
            Structure_aftertasks = Property()
        
        # BeforeGoal
        self.Significance = Significance
        self.State_of_the_art = State_of_the_art
        self.Gap = Gap
        self.Problem = Problem
        self.References = References
        # Goal
        self.Purpose = Purpose
        self.Intention = Intention
        self.Structure_goal = Structure_goal
        self.Congruence = Congruence
        # Tasks
        self.Outlook = Outlook
        self.Quantity = Quantity
        self.Completeness = Completeness
        self.Format = Format
        self.Structure_tasks = Structure_tasks
        self.Clarity = Clarity
        self.Relevance = Relevance
        # AfterTasks
        self.Chapters = Chapters
        self.Description = Description
        self.Structure_aftertasks = Structure_aftertasks

from src.code.parsing.old.Part import *
from src.code.parsing.old.Property import Property

class BeforeGoal(Part):

    def __init__(self, Significance : Property | None = None, State_of_the_art : Property | None = None, Gap : Property | None = None, Problem : Property | None = None, References : Property | None = None):
        
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

        self.Significance = Significance
        self.State_of_the_art = State_of_the_art
        self.Gap = Gap
        self.Problem = Problem
        self.References = References

        self.questions_mapping = {
            "Significance" : "Does the text clearly explain why the topic is important and relevant, and why it is necessary to research it in the context of current problems, trends, or developments?",
            "State_of_the_art" : "Does the text demonstrate awareness of the current state-of-the-art of the field by briefly citing specific, published research and mentioning existing cutting-edge achievements related to the topic? If the challenge the thesis is trying to address is evaluation or comparison of something then the state-of-the-art part should cite evaluations or comparisons that have been done before by other authors or state that no such results have yet been published (which then can be viewed as gap). If the challenge is development of new a solution, then the state-of-the-art part should cite existing solutions or state that no such solutions exist (which then can be viewed as gap).",
            "Gap" : "Does the text explicitly identify a concrete research gap, formulated on the basis of existing studies, by showing what remains unresolved or insufficiently addressed in the current state of knowledge or practice?",
            "Problem" : "Can it be easily identified what concrete issue or challenge the thesis is trying to address?",
            "References" : "Are references included where needed to support claims, acknowledge sources, or show how prior research is used, interpreted, or extended?"
        }
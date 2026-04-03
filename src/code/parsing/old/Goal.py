from src.code.parsing.old.Part import *
from src.code.parsing.old.Property import Property

class Goal(Part):

    def __init__(self, Purpose : Property | None = None, Intention : Property | None = None, Structure : Property | None = None, Congruence : Property | None = None):
        
        if Purpose is None:
            Purpose = Property()
        if Intention is None:
            Intention = Property()
        if Structure is None:
            Structure = Property()
        if Congruence is None:
            Congruence = Property()
        
        self.Purpose = Purpose
        self.Intention = Intention
        self.Structure = Structure
        self.Congruence = Congruence

        self.questions_mapping = {
            "Purpose" : "Does the goal clearly state what the student aims to accomplish in the thesis? Its wording should not be too general and avoid vague phrases. It should be possible to measure whether the goal has been reached.",
            "Intention" : "Does the goal focus on the intended outcome or objective, without describing methods, tools, or processes?",
            "Structure" : "Is the text limited to a single, concise sentence that directly communicates the goal of the thesis?",
            "Congruence" : "Does the goal build logically and consistently on the previously stated gaps, problems, or motivations in the preceding text, ensuring that the proposed goal directly respond to them? All aspects mentioned in the goal should be also mentioned in the preceding text."
        }
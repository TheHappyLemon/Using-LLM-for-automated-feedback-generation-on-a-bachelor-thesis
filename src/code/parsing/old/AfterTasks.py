from src.code.parsing.old.Part import *
from src.code.parsing.old.Property import Property

class AfterTasks(Part):

    def __init__(self, Chapters : Property | None = None, Description : Property | None = None, Structure : Property | None = None):
        if Chapters is None:
            Chapters = Property()
        if Description is None:
            Description = Property()
        if Structure is None:
            Structure = Property()
        
        self.Chapters = Chapters
        self.Description = Description
        self.Structure = Structure

        self.questions_mapping = {
            "Chapters" : "Does the text list the main sections of the thesis?",
            "Description" : " Does the text briefly describe contents of each section?",
            "Structure" : "Can it be easily understood what is the general structure of the thesis just from this text?"
        }

    def fully_complies(self):
        return self.Chapters.value == 1 and self.Description.value == 1 and self.Structure.value == 1

    def fully_not_complies(self):
        return self.Chapters.value == 0 and self.Description.value == 0 and self.Structure.value == 0
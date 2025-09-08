class PersonalityQuizTemplate:
    """Templates for formatting personality quiz questions."""
    
    @staticmethod
    def format_question(statement: str) -> str:
        """Format a personality statement into a Likert scale question.
        
        Args:
            statement: The personality statement to be rated
            
        Returns:
            Formatted question with Likert scale instructions
        """
        return f"""Rate how much you agree with this statement on a scale of 1-5:
1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree

Statement: {statement}"""


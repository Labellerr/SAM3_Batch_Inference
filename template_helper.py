import uuid
import random
from labellerr.core.annotation_templates import create_template
from labellerr.core.schemas.annotation_templates import AnnotationQuestion, QuestionType, CreateTemplateParams, DatasetDataType


def generate_random_color():
    """Generate a random hex color code."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def create_questions_from_prompts(text_prompts, question_type=QuestionType.polygon, required=True):
    """
    Create a list of AnnotationQuestion objects from text prompts.
    
    Args:
        text_prompts: List of text strings to use as questions
        question_type: Type of annotation question (default: polygon)
        required: Whether questions are required (default: True)
    
    Returns:
        List of AnnotationQuestion objects
    """
    questions = []
    for idx, prompt in enumerate(text_prompts, start=1):
        question = AnnotationQuestion(
            question_number=idx,
            question=prompt,
            question_id=str(uuid.uuid4()),
            question_type=question_type,
            required=required,
            color=generate_random_color()
        )
        questions.append(question)
    return questions
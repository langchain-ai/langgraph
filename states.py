from typing import Any, List
from pydantic import BaseModel

class UserInputState(BaseModel):
    user_input: str

class QuestionState(BaseModel):
    questions: List[str]

class DataState(BaseModel):
    retrieved_data: Any
    can_answer: bool

class AnswerState(BaseModel):
    answer: str

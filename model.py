from pydantic import BaseModel


class PersonalityNote(BaseModel):
    Time_spent_Alone: float
    Stage_fear: float
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: float
    Friends_circle_size: float
    Post_frequency: float

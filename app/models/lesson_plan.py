from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from datetime import date, time
from enum import Enum

class GradeLevel(str, Enum):
    K = "K"
    GRADE_1 = "1"
    GRADE_2 = "2"
    GRADE_3 = "3"
    GRADE_4 = "4"
    GRADE_5 = "5"
    GRADE_6 = "6"
    GRADE_7 = "7"
    GRADE_8 = "8"
    GRADE_9 = "9"
    GRADE_10 = "10"
    GRADE_11 = "11"
    GRADE_12 = "12"

class Subject(str, Enum):
    PHYSICAL_EDUCATION = "Physical Education"
    HEALTH = "Health"
    DRIVERS_ED = "Driver's Education"

class Standard(BaseModel):
    code: str = Field(..., description="Standard code (e.g., 2.1.12.PGD.1)")
    description: str = Field(..., description="Standard description")
    type: str = Field("CCCS", description="Type of standard (e.g., CCCS, SLS)")

class SmartGoal(BaseModel):
    specific: str = Field(..., description="What specifically will be accomplished?")
    measurable: str = Field(..., description="How will progress be measured?")
    achievable: str = Field(..., description="Is this realistic with available resources?")
    relevant: str = Field(..., description="How does this align with broader goals?")
    time_bound: str = Field(..., description="What is the timeframe for achievement?")
    
class Objective(BaseModel):
    smart_goal: SmartGoal
    description: str = Field(..., description="Learning objective")
    assessment_criteria: str = Field(..., description="How the objective will be assessed")
    language_objective: Optional[str] = Field(None, description="Specific objective for ELL students")

class Activity(BaseModel):
    name: str
    description: str
    duration: int = Field(..., description="Duration in minutes")
    materials: List[str] = []
    grouping: Optional[str] = None
    modifications: Optional[Dict[str, str]] = Field(
        None,
        description="Specific modifications for different student groups"
    )
    teaching_phase: str = Field(
        ..., 
        description="Phase of instruction (e.g., Direct Instruction, Guided Practice)"
    )

class Assessment(BaseModel):
    type: str
    description: str
    criteria: List[str]
    tools: Optional[List[str]] = None
    modifications: Optional[Dict[str, str]] = None

class DifferentiationPlan(BaseModel):
    ell_strategies: Dict[str, str] = Field(
        ...,
        description={
            "language_domains": "Reading, Writing, Speaking, Listening focus areas",
            "proficiency_level": "Student language development level",
            "strategies": "Specific ELL teaching strategies",
            "accommodations": "Language accommodations"
        }
    )
    iep_accommodations: Dict[str, str] = Field(
        ...,
        description="Specific accommodations for IEP students"
    )
    section_504_accommodations: Dict[str, str] = Field(
        ...,
        description="Accommodations for 504 students"
    )
    gifted_talented_enrichment: Dict[str, str] = Field(
        ...,
        description="Enrichment activities for gifted students"
    )

class LessonPlan(BaseModel):
    # Administrative Information
    teacher_name: str
    subject: Subject
    grade_level: GradeLevel
    unit_title: str
    lesson_title: str
    week_of: date
    date: date
    period: Optional[str] = None
    duration: int = Field(..., description="Lesson duration in minutes")

    # Standards and Objectives
    standards: List[Standard] = Field(..., description="All applicable standards")
    objectives: List[Objective] = Field(..., description="Learning objectives")

    # Lesson Components
    essential_question: str
    do_now: str = Field(..., description="Opening activity/warm-up")
    materials_needed: List[str]
    
    # Instructional Plan
    anticipatory_set: str = Field(
        ..., 
        description="Introduction/hook activity with specific statements/activities"
    )
    direct_instruction: str = Field(
        ...,
        description="Essential information and demonstration of skills"
    )
    guided_practice: List[Activity] = Field(
        ...,
        description="Teacher-guided practice activities"
    )
    independent_practice: List[Activity] = Field(
        ...,
        description="Student independent practice activities"
    )
    closure: str = Field(
        ...,
        description="Lesson conclusion, review, and evaluation method"
    )
    
    # Assessment and Differentiation
    assessments: List[Assessment]
    differentiation: DifferentiationPlan
    
    # Additional Components
    homework: Optional[str] = None
    notes: Optional[str] = None
    reflection: Optional[str] = None
    next_steps: Optional[str] = Field(
        None,
        description="Planning notes for future lessons based on this lesson's outcomes"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "teacher_name": "John Doe",
                "subject": "Physical Education",
                "grade_level": "9",
                "unit_title": "Basketball Fundamentals",
                "lesson_title": "Dribbling Techniques",
                "week_of": "2024-03-11",
                "date": "2024-03-15",
                "period": "3",
                "duration": 45,
                "standards": [{
                    "code": "2.1.12.PGD.1",
                    "description": "Develop a health care plan...",
                    "type": "CCCS"
                }],
                "objectives": [{
                    "smart_goal": {
                        "specific": "Master basic basketball dribbling techniques",
                        "measurable": "Complete dribbling course with 80% accuracy",
                        "achievable": "Progressive skill building with modifications",
                        "relevant": "Foundation for advanced basketball skills",
                        "time_bound": "By end of class period"
                    },
                    "description": "Students will demonstrate proper dribbling technique",
                    "assessment_criteria": "Successfully complete dribbling drill with 80% accuracy",
                    "language_objective": "Demonstrate understanding of dribbling terminology"
                }],
                "essential_question": "How does proper dribbling technique improve basketball performance?",
                "do_now": "Quick partner passing warm-up",
                "materials_needed": ["Basketballs", "Cones", "Whistle"],
                "anticipatory_set": "Quick dribbling demonstration and discussion",
                "direct_instruction": "Demonstration of proper dribbling technique with key points",
                "guided_practice": [{
                    "name": "Partner Dribbling Practice",
                    "description": "Students practice in pairs, giving feedback",
                    "duration": 15,
                    "materials": ["Basketballs"],
                    "grouping": "Pairs",
                    "teaching_phase": "Guided Practice",
                    "modifications": {
                        "ell": "Visual demonstrations",
                        "iep": "Modified equipment",
                        "gifted": "Complex patterns"
                    }
                }],
                "independent_practice": [{
                    "name": "Dribbling Course Challenge",
                    "description": "Individual completion of dribbling obstacle course",
                    "duration": 15,
                    "materials": ["Basketballs", "Cones"],
                    "teaching_phase": "Independent Practice"
                }],
                "closure": "Class demonstration and skill review",
                "assessments": [{
                    "type": "Performance",
                    "description": "Dribbling skills assessment",
                    "criteria": ["Proper hand position", "Head up while dribbling"],
                    "modifications": {
                        "ell": "Demonstration-based assessment",
                        "iep": "Modified success criteria"
                    }
                }],
                "differentiation": {
                    "ell_strategies": {
                        "language_domains": "Speaking, Listening",
                        "proficiency_level": "Intermediate",
                        "strategies": "Visual aids, demonstrations",
                        "accommodations": "Word wall, picture cards"
                    },
                    "iep_accommodations": {
                        "equipment": "Modified balls",
                        "pacing": "Extended practice time"
                    },
                    "section_504_accommodations": {
                        "physical": "Modified movement patterns",
                        "environmental": "Reduced noise level"
                    },
                    "gifted_talented_enrichment": {
                        "challenges": "Advanced dribbling patterns",
                        "leadership": "Peer coaching opportunities"
                    }
                },
                "homework": "Practice dribbling 15 minutes",
                "notes": "Focus on proper hand positioning",
                "reflection": "Students showed good progress",
                "next_steps": "Introduce crossover dribble next lesson"
            }
        }
    ) 

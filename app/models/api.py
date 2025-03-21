from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from enum import Enum

class MessageType(str, Enum):
    SMS = "sms"
    VOICE = "voice"

class DocumentRequest(BaseModel):
    document_type: str
    title: str
    content: str
    output_format: str = "docx"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_type": "report",
                "title": "Monthly Report",
                "content": "This is the content of the report",
                "output_format": "docx"
            }
        }
    )

class TextRequest(BaseModel):
    prompt: str
    structured_output: bool = False
    document_request: Optional[DocumentRequest] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Generate a summary of the quarterly results",
                "structured_output": False
            }
        }
    )

class TokenResponse(BaseModel):
    status: str
    token: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class UserInfoResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TextResponse(BaseModel):
    status: str
    content: Optional[str] = None
    error: Optional[str] = None

class SMSRequest(BaseModel):
    to_number: str = Field(..., description="Recipient's phone number in E.164 format (e.g., +1234567890)")
    message: str = Field(..., description="Message content to send")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "to_number": "+1234567890",
                "message": "Hello from FastAPI!"
            }
        }
    )

class SMSResponse(BaseModel):
    status: str
    message_sid: Optional[str] = None
    to: Optional[str] = None
    from_: Optional[str] = Field(None, alias="from")
    message_status: Optional[str] = None
    error: Optional[str] = None
    code: Optional[int] = None

class TranslationRequest(BaseModel):
    """Request model for text translation."""
    text: str
    source_language: str
    target_language: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Hello, how are you?",
                "source_language": "en",
                "target_language": "es"
            }
        }
    )

class TranslationResponse(BaseModel):
    """Response model for text translation."""
    translated_text: str
    status: str
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "translated_text": "Hola, ¿cómo estás?",
                "status": "success"
            }
        }
    )

class TranslatedMessageRequest(BaseModel):
    to_number: str = Field(..., description="Recipient's phone number in E.164 format")
    message: str = Field(..., description="Message to translate and send")
    message_type: MessageType = Field(MessageType.SMS, description="Type of message to send (sms or voice)")
    target_language: str = Field("es", description="Target language code")
    source_language: str = Field("en", description="Source language code")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "to_number": "+1234567890",
                "message": "Your child has performed exceptionally well in today's class.",
                "message_type": "sms",
                "target_language": "es",
                "source_language": "en"
            }
        }
    )

class TranslatedMessageResponse(BaseModel):
    status: str
    original_text: str
    translated_text: str
    message_type: str
    delivery_status: str
    message_sid: Optional[str] = None
    call_sid: Optional[str] = None
    error: Optional[str] = None

class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    name: Optional[str] = None

class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str
    context: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Hello, how are you?",
                "context": {}
            }
        }
    )

class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "I'm doing well, thank you!",
                "confidence": 0.95
            }
        }
    )

class ChatError(BaseModel):
    error: str
    details: Optional[Dict[str, Any]] = None

class LearningRequest(BaseModel):
    """Request model for learning content."""
    user_id: str
    topic: str
    difficulty: Optional[str] = "intermediate"
    preferred_style: Optional[str] = "interactive"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "topic": "Python Programming",
                "difficulty": "intermediate",
                "preferred_style": "interactive"
            }
        }
    )

class LearningResponse(BaseModel):
    """Response model for learning content."""
    content: Dict[str, Any]
    next_steps: Optional[List[str]] = None
    status: str
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_time: Optional[int] = None
    error: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": {
                    "summary": "Python is a high-level, interpreted programming language.",
                    "key_points": ["Python is easy to learn", "It's widely used in web development"]
                },
                "next_steps": ["Start with basic syntax", "Explore web frameworks"],
                "status": "success",
                "resources": [
                    {
                        "id": "res123",
                        "title": "Python for Beginners",
                        "description": "A beginner-friendly introduction to Python",
                        "url": "https://example.com/python-for-beginners",
                        "type": "article",
                        "difficulty": "beginner",
                        "estimated_time": 60
                    }
                ],
                "estimated_time": 120
            }
        }
    )

class ProgressUpdate(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    topic: str = Field(..., description="Topic being learned")
    status: str = Field(..., description="Current learning status")
    completion_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of completion")
    time_spent: int = Field(..., description="Time spent in minutes")
    achievements: List[str] = Field(default_factory=list, description="List of achievements earned")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "topic": "Python Programming",
                "status": "in_progress",
                "completion_percentage": 75.5,
                "time_spent": 45,
                "achievements": ["First Program", "Loop Master"]
            }
        }
    )

class ProgressResponse(BaseModel):
    status: str
    current_streak: Optional[int] = None
    total_time: Optional[int] = None
    achievements: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    error: Optional[str] = None

class ResourceType(str, Enum):
    VIDEO = "video"
    ARTICLE = "article"
    EXERCISE = "exercise"
    QUIZ = "quiz"
    TUTORIAL = "tutorial"
    PROJECT = "project"

class ResourceRequest(BaseModel):
    """Request model for educational resources."""
    user_id: str
    topic: str
    format: Optional[str] = None
    difficulty: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "topic": "Machine Learning",
                "format": "pdf",
                "difficulty": "intermediate"
            }
        }
    )

class ResourceMetadata(BaseModel):
    id: str
    title: str
    description: str
    url: Optional[str] = None
    type: ResourceType
    difficulty: float
    estimated_time: int  # in minutes
    prerequisites: List[str] = Field(default_factory=list)
    topics: List[str]
    engagement_score: Optional[float] = None
    completion_rate: Optional[float] = None

class ResourceListResponse(BaseModel):
    status: str
    resources: List[ResourceMetadata] = Field(default_factory=list)
    total_time: Optional[int] = None
    recommended_order: Optional[List[str]] = None
    error: Optional[str] = None

class ResourceResponse(BaseModel):
    """Response model for educational resources."""
    resources: List[Dict[str, Any]]
    total_count: int = Field(..., ge=0)
    status: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "resources": [
                    {
                        "id": "res123",
                        "title": "Introduction to Machine Learning",
                        "description": "A beginner-friendly tutorial on ML basics",
                        "url": "https://example.com/ml-intro",
                        "type": "tutorial",
                        "difficulty": 0.5,
                        "estimated_time": 60,
                        "prerequisites": ["Python Basics"],
                        "topics": ["Machine Learning", "Python"],
                        "engagement_score": 4.8,
                        "completion_rate": 0.92
                    }
                ],
                "total_count": 1,
                "status": "success"
            }
        }
    )

class LearningPathRequest(BaseModel):
    """Request model for learning path."""
    user_id: str
    current_topic: Optional[str] = None
    goals: Optional[List[str]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "current_topic": "Python Basics",
                "goals": ["Machine Learning"]
            }
        }
    )

class LearningPathStep(BaseModel):
    topic: str
    resources: List[ResourceMetadata]
    estimated_time: int  # in minutes
    prerequisites_met: bool
    completion_criteria: List[str]

class LearningPathResponse(BaseModel):
    """Response model for learning path."""
    user_id: str
    current_topics: List[str]
    recommended_topics: List[str]
    last_updated: str
    status: str
    path: List[LearningPathStep] = Field(default_factory=list)
    total_time: Optional[int] = None
    difficulty_curve: List[float] = Field(default_factory=list)
    milestones: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "current_topics": ["Python Basics"],
                "recommended_topics": ["Machine Learning"],
                "last_updated": "2024-04-01",
                "status": "success",
                "path": [
                    {
                        "topic": "Python Basics",
                        "resources": [],
                        "estimated_time": 120,
                        "prerequisites_met": True,
                        "completion_criteria": ["Complete basic syntax quiz"]
                    }
                ],
                "total_time": 600,
                "difficulty_curve": [0.3, 0.4, 0.6],
                "milestones": ["Complete Python Basics", "Master Data Structures"]
            }
        }
    )

class LearningProgressResponse(BaseModel):
    status: str = Field(..., description="Status of the request")
    summary: Dict[str, Any] = Field(..., description="User's learning progress summary")
    topic_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Recommended topics based on user's progress"
    )
    error: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "summary": {
                    "total_topics_completed": 5,
                    "current_streak": 7,
                    "average_completion_time": 45,
                    "strongest_areas": ["Python", "Data Structures"],
                    "areas_for_improvement": ["Algorithms"]
                },
                "topic_recommendations": [
                    {
                        "topic": "Advanced Algorithms",
                        "reason": "Based on your progress in Data Structures",
                        "difficulty": 0.8
                    }
                ]
            }
        }
    )

class DifficultyPredictionResponse(BaseModel):
    status: str
    difficulty_score: float
    confidence: float
    similar_content: List[Dict[str, Any]] = Field(default_factory=list)
    features: Dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "difficulty_score": 0.75,
                "confidence": 0.9,
                "similar_content": [
                    {
                        "title": "Advanced Python Concepts",
                        "difficulty": 0.8,
                        "similarity": 0.85
                    }
                ],
                "features": {
                    "complexity": 0.7,
                    "prerequisites_count": 0.6,
                    "concept_density": 0.8
                }
            }
        }
    )

class ProgressUpdateRequest(BaseModel):
    """Request model for progress updates."""
    user_id: str
    topic: str
    score: float
    time_spent: Optional[int] = None

class ProgressUpdateResponse(BaseModel):
    """Response model for progress updates."""
    user_id: str
    topic: str
    score: float
    streak: int
    achievements: List[str]

class ChallengeResponse(BaseModel):
    """Response model for daily challenges."""
    user_id: str
    date: str
    challenges: List[Dict[str, Any]]
    completed: List[int]

class AnalyticsResponse(BaseModel):
    """Response model for user analytics."""
    user_id: str
    metrics: Dict[str, Any]

class NotificationRequest(BaseModel):
    """Request model for notifications."""
    user_id: str
    message: str
    priority: Optional[str] = "normal"
    expiry: Optional[str] = None

class CalendarEventRequest(BaseModel):
    """Request model for calendar events."""
    user_id: str
    title: str
    start_time: str
    end_time: str
    description: Optional[str] = ""
    location: Optional[str] = ""

class LMSSyncResponse(BaseModel):
    """Response model for LMS synchronization."""
    user_id: str
    timestamp: str
    synced_data: Dict[str, Any]
    status: str 

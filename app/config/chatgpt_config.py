from openai import AsyncOpenAI
from loguru import logger
from app.config import get_settings
import httpx

class ChatGPTService:
    def __init__(self):
        logger.debug("Creating new ChatGPTService instance")
        self.settings = get_settings()
        logger.debug("Initializing ChatGPTService")
        
        try:
            logger.debug("Creating AsyncOpenAI client")
            # Initialize OpenAI client with just the API key
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            
            # Set default model
            self.model = "gpt-3.5-turbo"
            logger.debug("Successfully created AsyncOpenAI client")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    async def send_message(self, message: str) -> str:
        try:
            logger.debug(f"Sending message to OpenAI: {message}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": message}],
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )
            logger.debug("Received response from OpenAI")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in send_message: {str(e)}")
            raise

_chatgpt_service = None

def get_chatgpt_service() -> ChatGPTService:
    global _chatgpt_service
    if _chatgpt_service is None:
        _chatgpt_service = ChatGPTService()
    return _chatgpt_service

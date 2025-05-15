from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Configurations
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Tech Challenge 4 API"
    # Logging
    LOG_LEVEL: str = "info"
    # Environment
    ENV: str = "development"
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "models-bucket-tc4")

    class Config:
        env_file = ".env"

if os.getenv('ENV') == 'production':
    settings = Settings()
    """
    settings = Settings(
        JWT_SECRET=get_ssm_parameter("/my-fastApi-app/JWT_SECRET"),
        DATABASE_URL=get_ssm_parameter("/my-fastApi-app/DATABASE_URL"),
        BUCKET_NAME=get_ssm_parameter("/my-fastApi-app/BUCKET_NAME")
    )
    """
else:
    settings = Settings()
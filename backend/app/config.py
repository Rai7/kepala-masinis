from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    supabase_url: str
    supabase_service_role_key: str

    mongodb_uri: str | None = Field(default=None, validation_alias=AliasChoices("MONGODB_URI"))
    mongodb_db_name: str = Field(default="gapeka_chatbot", validation_alias=AliasChoices("MONGODB_DB_NAME"))
    enable_mongo_logging: bool = Field(
        default=False, validation_alias=AliasChoices("ENABLE_MONGO_LOGGING")
    )
    enable_mongo_cache: bool = Field(
        default=False, validation_alias=AliasChoices("ENABLE_MONGO_CACHE")
    )
    mongo_cache_ttl_seconds: int = Field(
        default=3600, validation_alias=AliasChoices("MONGO_CACHE_TTL_SECONDS")
    )

    tavily_api_key: str | None = Field(default=None, validation_alias=AliasChoices("TAVILY_API_KEY"))
    enable_tavily_search: bool = Field(
        default=False, validation_alias=AliasChoices("ENABLE_TAVILY_SEARCH")
    )
    tavily_max_results: int = Field(default=5, validation_alias=AliasChoices("TAVILY_MAX_RESULTS"))
    tavily_search_depth: str = Field(
        default="basic", validation_alias=AliasChoices("TAVILY_SEARCH_DEPTH")
    )

    open_weather_api_key: str | None = Field(default=None, validation_alias=AliasChoices("OPEN_WEATHER_API_KEY", "OPENWEATHER_API_KEY"))
    enable_open_weather: bool = Field(
        default=True, validation_alias=AliasChoices("ENABLE_OPEN_WEATHER")
    )

    llm_enabled: bool = Field(default=True, validation_alias=AliasChoices("LLM_ENABLED"))
    llm_response_formatting: bool = Field(
        default=True, validation_alias=AliasChoices("LLM_RESPONSE_FORMATTING")
    )

    groq_api_key: str | None = Field(default=None, validation_alias=AliasChoices("GROQ_API_KEY"))
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", validation_alias=AliasChoices("GROQ_BASE_URL"))
    groq_model: str | None = Field(default=None, validation_alias=AliasChoices("GROQ_MODEL"))

    featherless_api_key: str | None = None
    featherless_base_url: str = "https://api.featherless.ai/v1"
    featherless_model: str | None = Field(
        default=None, validation_alias=AliasChoices("FEATHERLESS_MODEL", "MODEL_NAME")
    )

    cors_allow_origins: str = "*"


settings = Settings()

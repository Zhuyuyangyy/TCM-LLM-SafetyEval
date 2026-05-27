from dataclasses import dataclass

@dataclass
class Settings:
    app_name: str = "TCM-LLM-SafetyEval"
    version: str = "0.3.0"
    port: int = 8029

settings = Settings()

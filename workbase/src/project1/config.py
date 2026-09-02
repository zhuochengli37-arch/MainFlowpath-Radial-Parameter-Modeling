from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False


@dataclass(frozen=True)
class Settings:
    app_name: str = "project1"
    log_level: str = "INFO"
    data_dir: str = "./data"
    log_file: str = "./data/output/app.log"


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        app_name=os.getenv("APP_NAME", "project1"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        data_dir=os.getenv("DATA_DIR", "./data"),
        log_file=os.getenv("LOG_FILE", "./data/output/app.log"),
    )

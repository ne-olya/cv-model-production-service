from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISION_", env_file=".env", extra="ignore")
    model_path: str | None = None
    model_version: str = "demo-1.0.0"
    device: str = "cpu"
    image_size: int = 224
    max_file_mb: int = 10
    max_batch_size: int = 16
    confidence_threshold: float = 0.0
    classes: str = "plastic,glass,paper,metal,cardboard,organic,other"

    @property
    def class_names(self):
        return [value.strip() for value in self.classes.split(",") if value.strip()]

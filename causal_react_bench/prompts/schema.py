from dataclasses import dataclass


@dataclass
class SourceRecord:
    src_set: str
    src_vid: str
    prompt: str
    seconds: float


@dataclass
class Candidate:
    prompt: str


@dataclass
class ReasonResult:
    valid: bool
    prompts: list[Candidate]
    reason: str


@dataclass
class ReviewResult:
    valid: bool
    faithful: bool
    single_scene: bool
    physically_plausible: bool
    fits_duration: bool
    prompt: str
    reason: str

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewResult":
        if not isinstance(data, dict):
            raise ValueError("Review response must be a JSON object")
        flags = ("valid", "faithful", "single_scene", "physically_plausible", "fits_duration")
        for key in flags:
            if type(data.get(key)) is not bool:
                raise ValueError(f"Review field {key!r} must be a boolean")
        for key in ("prompt", "reason"):
            if not isinstance(data.get(key), str):
                raise ValueError(f"Review field {key!r} must be a string")
        return cls(**{key: data[key] for key in (*flags, "prompt", "reason")})

    @property
    def accepted(self) -> bool:
        return all((self.valid, self.faithful, self.single_scene,
                    self.physically_plausible, self.fits_duration)) and bool(self.prompt.strip())

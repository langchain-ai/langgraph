from __future__ import annotations

from dataclasses import dataclass, replace

DIGEST_SEPARATOR = "@sha256:"
DIGEST_MARKER = "@"
TAG_SEPARATOR = ":"
PATH_SEPARATOR = "/"


@dataclass(frozen=True, slots=True)
class ImageReference:
    repository: str
    tag: str | None = None

    @classmethod
    def parse(cls, reference: str) -> ImageReference:
        if DIGEST_MARKER in reference:
            raise ValueError(f"{reference!r} carries a digest and cannot be tagged")
        path_start = reference.rfind(PATH_SEPARATOR) + 1
        name, separator, tag = reference[path_start:].partition(TAG_SEPARATOR)
        if not separator:
            return cls(reference)
        return cls(reference[:path_start] + name, tag)

    def with_tag(self, tag: str) -> ImageReference:
        return replace(self, tag=tag)

    def matches_digest(self, repo_digest: str) -> bool:
        return repo_digest.startswith(f"{self.repository}{DIGEST_SEPARATOR}")

    def __str__(self) -> str:
        if self.tag is None:
            return self.repository
        return f"{self.repository}{TAG_SEPARATOR}{self.tag}"

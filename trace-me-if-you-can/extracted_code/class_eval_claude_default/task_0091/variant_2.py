import urllib.parse
from typing import List


class UrlPath:
    def __init__(self):
        self.segments: List[str] = []
        self.with_end_tag: bool = False

    def add(self, segment: str) -> None:
        self.segments.append(self._normalize_segment(segment))

    def parse(self, path: str, charset: str) -> None:
        if path:
            self._set_end_tag_flag(path)
            self._extract_segments(path, charset)

    def _set_end_tag_flag(self, path: str) -> None:
        self.with_end_tag = path.endswith('/')

    def _extract_segments(self, path: str, charset: str) -> None:
        normalized_path = self._normalize_segment(path)
        if normalized_path:
            raw_segments = normalized_path.split('/')
            decoded_segments = [urllib.parse.unquote(seg, encoding=charset) for seg in raw_segments]
            self.segments.extend(decoded_segments)

    def _normalize_segment(self, path: str) -> str:
        return path.strip('/') if path else ''

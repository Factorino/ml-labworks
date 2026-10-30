from dataclasses import dataclass
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, override
import unicodedata
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from mllab import DATA_DIR


USER_AGENT = "mllab/0.1 (https://github.com/Factorino/ml-labworks)"
WIKISOURCE_API = "https://ru.wikisource.org/w/api.php"
GUTENBERG = "https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt"

# fmt: off
CHAPTERS = [
    "I", "II", "III", "IV", "V", "VI", "VII",
    "VIII", "IX", "X", "XI", "XII", "XIII", "XIV",
]
# fmt: on
BELKIN = ["Выстрел", "Метель", "Гробовщик", "Станционный смотритель", "Барышня-крестьянка"]
RUSSIAN_PAGES = [
    *(f"Капитанская дочка (Пушкин)/1960 (СО)/Глава {chapter}" for chapter in CHAPTERS),
    *(f"Повести покойного Ивана Петровича Белкина (Пушкин)/{tale}" for tale in BELKIN),
]

TRANSLATION = 55219
TRANSLATION_TITLES = [
    "THE CAPTAIN’S DAUGHTER.",
    "DOUBROVSKY.",
    "THE QUEEN OF SPADES.",
    "AN AMATEUR PEASANT GIRL.",
    "THE SHOT.",
    "THE SNOWSTORM.",
    "THE POSTMASTER.",
    "THE COFFIN-MAKER.",
    "KIRDJALI.",
    "THE EGYPTIAN NIGHTS",
    "PETER THE GREAT’S NEGRO.[1]",
]
ENGLISH_WORKS = [
    "THE CAPTAIN’S DAUGHTER.",
    "THE SHOT.",
    "THE SNOWSTORM.",
    "THE COFFIN-MAKER.",
    "THE POSTMASTER.",
    "AN AMATEUR PEASANT GIRL.",
]

PUNCTUATION = " \n.,;:!?-—()'\"«»"
ALPHABETS = {
    "ru": "абвгдежзийклмнопрстуфхцчшщъыьэюя" + PUNCTUATION,
    "en": "abcdefghijklmnopqrstuvwxyz" + PUNCTUATION,
}
REPLACEMENTS = {
    "ru": {"ё": "е", "„": "«", "“": "»", "…": "...", "‑": "-"},
    "en": {"“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "‑": "-"},
}


def download(url: str) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT})) as response:  # noqa: S310
        body: bytes = response.read()
    return body


class ProseParser(HTMLParser):
    VOID_TAGS = frozenset({"br", "hr", "img", "input", "link", "meta", "wbr"})
    SKIPPED_TAGS = frozenset({"sup", "style", "h2", "h3", "h4"})
    SKIPPED_CLASSES = ("ws-noexport", "license", "epigraph", "poem", "references", "mw-heading")

    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self.stack: list[tuple[str, bool]] = []
        self.current: list[str] | None = None

    @property
    def skipping(self) -> bool:
        return any(skipped for _, skipped in self.stack)

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.VOID_TAGS:
            return

        classes = dict(attrs).get("class") or ""
        skipped = tag in self.SKIPPED_TAGS or any(name in classes for name in self.SKIPPED_CLASSES)
        self.stack.append((tag, skipped))
        if tag == "p" and not self.skipping:
            self.current = []

    @override
    def handle_endtag(self, tag: str) -> None:
        tags = [opened for opened, _ in self.stack]
        if tag not in tags:
            return

        del self.stack[len(tags) - 1 - tags[::-1].index(tag) :]

        if tag == "p" and self.current is not None:
            self.paragraphs.append("".join(self.current))
            self.current = None

    @override
    def handle_data(self, data: str) -> None:
        if self.current is not None and not self.skipping:
            self.current.append(data)


def wikisource_paragraphs(page: str) -> list[str]:
    query = {"action": "parse", "page": page, "prop": "text", "format": "json", "formatversion": 2}
    html = json.loads(download(f"{WIKISOURCE_API}?{urlencode(query)}"))["parse"]["text"]
    parser = ProseParser()
    parser.feed(html)
    return parser.paragraphs


def gutenberg_paragraphs(book: str, titles: list[str], works: list[str]) -> list[str]:
    body = book.split("*** START OF")[1].split("*** END OF", maxsplit=1)[0]
    found: dict[str, list[str]] = {}
    current: list[str] | None = None

    for block in re.split(r"\n\s*\n", body):
        line = " ".join(block.split())
        if line in titles:
            current = found.setdefault(line, [])

        elif line.isupper() or line.startswith("[Footnote"):
            continue  # chapter headings, the headers of letters, footnotes

        elif current is not None and line:
            current.append(re.sub(r"\[\d+\]|_", "", line))

    return [paragraph for work in works for paragraph in found[work]]


def cached_text(path: Path, fetch: Any) -> str:
    if not path.exists():
        paragraphs: list[str] = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(" ".join(p.split()) for p in paragraphs if p.strip()), "utf-8")
    return path.read_text(encoding="utf-8")


def load_russian() -> str:
    return cached_text(
        DATA_DIR / "pushkin_ru.txt",
        lambda: [p for page in RUSSIAN_PAGES for p in wikisource_paragraphs(page)],
    )


def load_english() -> str:
    return cached_text(
        DATA_DIR / "pushkin_en.txt",
        lambda: gutenberg_paragraphs(
            download(GUTENBERG.format(book=TRANSLATION)).decode("utf-8"),
            TRANSLATION_TITLES,
            ENGLISH_WORKS,
        ),
    )


def normalize(text: str, language: str) -> str:
    text = text.lower()
    for old, new in REPLACEMENTS[language].items():
        text = text.replace(old, new)

    if language == "en":
        decomposed = unicodedata.normalize("NFD", text)
        text = "".join(c for c in decomposed if not unicodedata.combining(c))
        text = re.sub(r"-{2,}", "—", text)

    alphabet = set(ALPHABETS[language])
    text = "".join(c for c in text if c in alphabet)
    lines = (" ".join(line.split()) for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class Vocabulary:
    chars: str

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        return cls("".join(sorted(set(text))))

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> NDArray[np.int32]:
        index = {char: code for code, char in enumerate(self.chars)}
        return np.array([index[char] for char in text], dtype=np.int32)

    def decode(self, codes: Any) -> str:
        return "".join(self.chars[code] for code in codes)


def make_windows(
    codes: NDArray[np.int32], length: int, step: int = 1
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    windows = sliding_window_view(codes[:-1], length)[::step]
    return np.ascontiguousarray(windows), codes[length::step]


def sample(
    probabilities: NDArray[np.floating[Any]], temperature: float, rng: np.random.Generator
) -> int:
    if temperature == 0:
        return int(np.argmax(probabilities))

    logits = np.log(np.maximum(probabilities.astype(np.float64), 1e-12)) / temperature
    weights = np.exp(logits - logits.max())
    return int(rng.choice(len(weights), p=weights / weights.sum()))


def generate(
    model: Any,
    vocabulary: Vocabulary,
    seed: str,
    length: int,
    *,
    temperature: float,
    rng: np.random.Generator,
) -> str:
    window = model.input_shape[1]
    codes = list(vocabulary.encode(seed))

    for _ in range(length):
        context = np.array([codes[-window:]], dtype=np.int32)
        probabilities = np.asarray(model(context, training=False))[0]
        codes.append(sample(probabilities, temperature, rng))

    return vocabulary.decode(codes[len(seed) :])

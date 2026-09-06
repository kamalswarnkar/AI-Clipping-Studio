"""Repairing names the transcript itself disagrees about.

Speech recognition drops leading syllables: the same rally transcript says
"Antifa" three times and "Tifa" once. Downstream that is worse than a plain
mistake, because the summary sees two spellings and reports two entities --
"Participants: Nick Shirley, Tifa, ... and members of Antifa" -- and every hook
and caption written from it inherits the invented person.

The repair uses only evidence already in the transcript: a rare token that is
the *tail* of a common one is a truncation of it. No world knowledge, so no
opportunity to invent. Asking a 7B model to correct proper nouns instead was
tried and rejected -- it turned "Tifa" into "Tia" and "Nick Shirley" into
"Nickolas J. Shirley".
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from ..models.domain import Transcript

log = logging.getLogger(__name__)

# Short forms below this length are too ambiguous to merge on a suffix match.
MIN_LENGTH = 4
# The long form has to be this much more common before it wins.
DOMINANCE = 2.0
# And it has to be established, not itself a one-off.
MIN_LONG_COUNT = 2

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
# A word that follows one of these is sentence-initial, so its capital says
# nothing about whether it is a name.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s*$")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def proper_nouns(transcript: Transcript) -> set[str]:
    """Tokens the transcript capitalises somewhere other than after a full stop.

    This is the guard that keeps the suffix rule off ordinary English. Without
    it the same rule happily proposes "public" -> "republic", "rally" ->
    "literally" and "ever" -> "never": all real suffix matches, all nonsense.
    A name is a name because the transcript capitalises it mid-sentence.
    """
    names: set[str] = set()
    for segment in transcript.segments:
        previous = ""
        for token in _WORD.finditer(segment.text):
            word = token.group()
            starts_sentence = not previous or bool(
                _SENTENCE_END.search(segment.text[: token.start()].rstrip())
            )
            if word[:1].isupper() and not starts_sentence:
                names.add(word.lower())
            previous = word
    return names


def find_truncations(transcript: Transcript) -> dict[str, str]:
    """Map rare truncated spellings to the full form the transcript prefers.

    Only suffix relationships count. Recognition mishears the *start* of a word
    it does not know ("Antifa" -> "Tifa"), so a short form that matches the end
    of a longer one is the case worth merging; matching the start is far more
    likely to be two genuinely different words ("Nick" and "Nickolas").
    """
    counts = Counter(token.lower() for token in _tokens(transcript.text))
    if not counts:
        return {}

    names = proper_nouns(transcript)
    mapping: dict[str, str] = {}
    for short, short_count in counts.items():
        if len(short) < MIN_LENGTH or short not in names:
            continue
        best: str | None = None
        for long, long_count in counts.items():
            if len(long) <= len(short) or not long.endswith(short):
                continue
            if long not in names:
                continue
            if long_count < MIN_LONG_COUNT or long_count < short_count * DOMINANCE:
                continue
            # Prefer the shortest qualifying long form: "antifa" over
            # "antifascist" when both are present.
            if best is None or len(long) < len(best):
                best = long
        if best:
            mapping[short] = best
            log.info(
                "Transcript truncation: %r (x%d) -> %r (x%d)",
                short,
                short_count,
                best,
                counts[best],
            )
    return mapping


def _match_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def apply_truncations(transcript: Transcript, mapping: dict[str, str]) -> int:
    """Rewrite the truncated spellings in place. Returns words changed.

    Word timings are untouched -- only the text of a word changes -- so burned-in
    captions stay in sync.
    """
    if not mapping:
        return 0

    changed = 0
    for segment in transcript.segments:
        for word in segment.words:
            bare = _WORD.search(word.word)
            if not bare:
                continue
            replacement = mapping.get(bare.group().lower())
            if not replacement:
                continue
            fixed = _match_case(replacement, bare.group())
            word.word = word.word.replace(bare.group(), fixed, 1)
            changed += 1

        def swap(match: re.Match[str]) -> str:
            replacement = mapping.get(match.group().lower())
            return _match_case(replacement, match.group()) if replacement else match.group()

        segment.text = _WORD.sub(swap, segment.text)

    return changed


def apply_to_text(text: str, mapping: dict[str, str]) -> str:
    """Rewrite truncated spellings in free text.

    The video-level summary is built from the transcript *before* this repair
    runs, so it carries the truncated spelling into every clip description,
    hook and caption written from it -- which is how an invented participant
    called "Tifa" reached a published caption. Rewriting the summary is far
    cheaper than summarising the video again.
    """
    if not text or not mapping:
        return text

    def swap(match: "re.Match[str]") -> str:
        replacement = mapping.get(match.group().lower())
        return _match_case(replacement, match.group()) if replacement else match.group()

    return _WORD.sub(swap, text)

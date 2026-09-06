import re

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

from app.config import DEFAULT_CHUNK_SIZE, MAX_DEFAULT_CHUNK_OVERLAP

def _split_paragraphs(text):
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]


def _split_sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _split_words(text):
    return text.split()


def _hard_token_cut(text, tokenizer, size):
    """Last-resort fallback for a single whitespace-delimited 'word' that
    is itself too large in tokens to fit even alone (e.g. a long reference
    code or URL with no internal spaces for _split_words to break on).

    Cuts directly at the token level so the size ceiling is never violated,
    no matter how the text is structured. No overlap is applied here,
    consistent with how oversized units are already handled elsewhere in
    _pack — a clean break, not a merge.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(ids[i:i + size]) for i in range(0, len(ids), size)]


def _pack(units, tokenizer, size, overlap, fallback, join_sep):
    """Greedily pack `units` into chunks of up to `size` tokens, carrying
    up to `overlap` tokens of trailing units into the next chunk.
    A unit that alone exceeds `size` tokens can't be packed at all -- it's
    handed to `fallback(unit_text)` instead, which must return a list of
    already-sized chunk strings for that unit. `fallback=None` means there
    is nothing finer to fall back to (word level); such a unit is emitted
    as its own oversized chunk.
    """
    counted = [(u, len(tokenizer.encode(u, add_special_tokens=False))) for u in units]
    chunks, current, current_tokens = [], [], 0
    for unit, count in counted:
        if count > size:
            if current:
                chunks.append(join_sep.join(u for u, _ in current))
                current, current_tokens = [], 0
            chunks.extend(fallback(unit) if fallback else [unit])
            continue
        if current and current_tokens + count > size:
            chunks.append(join_sep.join(u for u, _ in current))
            kept, kept_tokens = [], 0
            for u, c in reversed(current):
                if kept_tokens + c > overlap:
                    break
                kept.insert(0, (u, c))
                kept_tokens += c
            current, current_tokens = kept, kept_tokens
            if current and current_tokens + count > size:
                current, current_tokens = [], 0
        current.append((unit, count))
        current_tokens += count
    if current:
        chunks.append(join_sep.join(u for u, _ in current))
    return chunks


def chunk_text(text, tokenizer, size=DEFAULT_CHUNK_SIZE, overlap=MAX_DEFAULT_CHUNK_OVERLAP):
    """Chunk `text` by falling back through progressively finer boundaries:
    paragraph -> sentence -> word -> raw token cut. A paragraph that
    already fits becomes a single chunk untouched; only oversized
    paragraphs get split further, and only oversized sentences within
    them fall through to word-level, with a hard token-level cut as the
    final safety net for any single unsplittable word.
    """
    if size <= overlap:
        raise ValueError(f"chunk size ({size}) must be greater than overlap ({overlap})")

    def hard_cut(unit_text):
        return _hard_token_cut(unit_text, tokenizer, size)

    def word_level(unit_text):
        return _pack(_split_words(unit_text), tokenizer, size, overlap, fallback=hard_cut, join_sep=" ")

    def sentence_level(unit_text):
        return _pack(_split_sentences(unit_text), tokenizer, size, overlap, fallback=word_level, join_sep=" ")

    chunks = []
    for paragraph in _split_paragraphs(text):
        token_count = len(tokenizer.encode(paragraph, add_special_tokens=False))
        if token_count <= size:
            chunks.append(paragraph)
        else:
            chunks.extend(sentence_level(paragraph))
    return chunks
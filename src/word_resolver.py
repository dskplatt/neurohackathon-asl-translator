import math


class WordResolver:
    def __init__(self, dictionary_path=None):
        if dictionary_path is not None:
            with open(dictionary_path, "r") as f:
                word_list = [line.strip() for line in f]
        else:
            import ssl
            import nltk
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            nltk.download("words", quiet=True)
            from nltk.corpus import words
            word_list = words.words()

        from wordfreq import word_frequency

        # Common single-letter words (NLTK may not include them)
        single_letter_words = {"a", "i"}

        filtered = [
            w.lower()
            for w in word_list
            if w.isalpha() and 1 <= len(w) <= 15
        ]
        filtered = sorted(set(filtered) | single_letter_words)

        self.freq_cache: dict[str, float] = {}
        self.by_length: dict[int, list[str]] = {}

        for word in filtered:
            freq = word_frequency(word, "en")
            self.freq_cache[word] = freq if freq > 0 else 1e-9

            n = len(word)
            if n not in self.by_length:
                self.by_length[n] = []
            self.by_length[n].append(word)

        print(
            f"WordResolver loaded {len(filtered)} words "
            f"across {len(self.by_length)} lengths"
        )

    def score_word(self, word: str, letter_distributions: list[dict]) -> float:
        if len(word) != len(letter_distributions):
            return 0.0

        score = 1.0
        for i, ch in enumerate(word.lower()):
            p = letter_distributions[i].get(ch, 1e-9)
            score *= max(p, 1e-9)

        freq = self.freq_cache.get(word.lower(), 1e-9)
        score *= math.log(freq + 1)

        return score

    def resolve(
        self, letter_distributions: list[dict], top_n: int = 5
    ) -> list[tuple[str, float]]:
        n = len(letter_distributions)
        candidates = self.by_length.get(n, [])

        if not candidates:
            return [("?", 0.0)]

        scored = [(w, self.score_word(w, letter_distributions)) for w in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_n]


if __name__ == "__main__":
    resolver = WordResolver()

    # Test 1 — HELLO with uncertain E
    distributions = [
        {c: (0.91 if c == "h" else 0.01) for c in "abcdefghijklmnopqrstuvwxyz"},
        {c: (0.40 if c == "e" else 0.40 if c == "a" else 0.01) for c in "abcdefghijklmnopqrstuvwxyz"},
        {c: (0.95 if c == "l" else 0.01) for c in "abcdefghijklmnopqrstuvwxyz"},
        {c: (0.88 if c == "l" else 0.01) for c in "abcdefghijklmnopqrstuvwxyz"},
        {c: (0.91 if c == "o" else 0.01) for c in "abcdefghijklmnopqrstuvwxyz"},
    ]
    results = resolver.resolve(distributions, top_n=5)
    print("Test 1 — HELLO with uncertain E:")
    for word, score in results:
        print(f"  {word}: {score:.6f}")
    assert results[0][0] == "hello", f"FAIL: expected 'hello', got '{results[0][0]}'"
    print("  PASS")

    # Test 2 — SHORT word (3 letters), clear signals
    distributions = [
        {c: (0.95 if c == "c" else 0.01) for c in "abcdefghijklmnopqrstuvwxyz"},
        {c: (0.92 if c == "a" else 0.01) for c in "abcdefghijklmnopqrstuvwxyz"},
        {c: (0.88 if c == "t" else 0.01) for c in "abcdefghijklmnopqrstuvwxyz"},
    ]
    results = resolver.resolve(distributions, top_n=5)
    print("Test 2 — CAT:")
    for word, score in results:
        print(f"  {word}: {score:.6f}")
    assert results[0][0] == "cat", f"FAIL: expected 'cat', got '{results[0][0]}'"
    print("  PASS")

    # Test 3 — Score never returns exactly 0.0
    dist = [{c: 0.0 for c in "abcdefghijklmnopqrstuvwxyz"} for _ in range(5)]
    score = resolver.score_word("hello", dist)
    assert score > 0.0, "FAIL: score returned 0.0"
    print(f"Test 3 — Zero-probability floor: score={score:.2e}  PASS")

    # Test 4 — resolve() returns correct types
    results = resolver.resolve(distributions, top_n=3)
    assert isinstance(results, list), "FAIL: not a list"
    assert len(results) == 3, f"FAIL: expected 3 results, got {len(results)}"
    assert all(isinstance(w, str) for w, _ in results), "FAIL: words not strings"
    assert all(isinstance(s, float) for _, s in results), "FAIL: scores not floats"
    assert all(w == w.lower() for w, _ in results), "FAIL: words not lowercase"
    print("Test 4 — Return types: PASS")

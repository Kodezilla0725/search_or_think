"""Answer scoring (SQuAD / HotpotQA style EM and F1) and evidence matching."""
import collections
import re
import string
import unicodedata

_PUNCT = set(string.punctuation)


def normalize_answer(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, golds):
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def f1(pred, golds):
    p = normalize_answer(pred)
    best = 0.0
    for g in golds:
        g = normalize_answer(g)
        # HotpotQA official eval: yes/no answers only score on an exact match
        if p != g and ({p, g} & {"yes", "no", "noanswer"}):
            continue
        common = collections.Counter(p.split()) & collections.Counter(g.split())
        ns = sum(common.values())
        if ns:
            prec, rec = ns / len(p.split()), ns / len(g.split())
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def has_answer(text, golds):
    t = f" {normalize_answer(text)} "
    return any(f" {normalize_answer(g)} " in t for g in golds if normalize_answer(g))


def norm_title(t):
    return unicodedata.normalize("NFKC", t or "").strip().strip('"').lower()


def norm_query(q):
    """Two queries that differ only in case or spacing are the same query: BM25 returns the same passages."""
    return " ".join((q or "").lower().split())


def n_distinct(queries):
    """Distinct queries in one episode. A repeated query retrieves nothing new, so this is how many retrievals
    the episode really used; N_min is estimated from it (see analysis.boundary_table)."""
    return len({norm_query(q) for q in queries})


def knows(answer, golds):
    """The correctness test behind p(q): cover match, i.e. a gold answer is contained in the closed book answer.
    Lenient on form because p(q) measures knowledge, and strict EM misses verbose correct answers (interim report)."""
    return has_answer(answer or "", golds)
"""Prompts and the batched multi turn search agent loop (model and retriever passed in as callables)."""
import re

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S)
ANSWER_RE = re.compile(r"<answer>((?:(?!<answer>).)*?)</answer>", re.S)  # innermost, if a prefill <answer> is repeated
LIMIT_MSG = "\n\n<info>\nSearch limit reached. Give your final answer now.\n</info>\n\n"
# "<info>" is a stop string: only the environment may write search results. A model that opens its own <info> ends
# the episode there (stop_reason "fake_info"), scored on whatever answer it has, which is normally none.
STOP = ["</search>", "</answer>", "<info>"]

_ANSWER_FMT = ("give the final answer inside <answer> and </answer> using only a few words, "
               "for example <answer> Paris </answer>.")
_DIRECT_FMT = "Respond with only the final answer in a few words inside <answer> and </answer>, with no explanation."

_AGENT_EXAMPLE = ("Example:\nQuestion: What is the capital of the country where the Eiffel Tower is?\n"
                  "<think> I need to know which country the Eiffel Tower is in. </think>\n"
                  "<search> Eiffel Tower location </search>\n\n"
                  "<info>\nDoc 1 (Title: Eiffel Tower) The Eiffel Tower is a wrought-iron lattice tower in Paris, France.\n</info>\n\n"
                  "<think> The tower is in France, and the capital of France is Paris. </think>\n"
                  "<answer> Paris </answer>")


def agent_instruction(question, max_searches=4):
    return ("Answer the question below. Reason inside <think> and </think> every time you get new information. "
            "If you lack some knowledge, search Wikipedia by writing <search> your query </search>, and the top "
            "results will be returned between <info> and </info>. Never invent search results. You can search up to "
            f"{max_searches} times. When you are ready, {_ANSWER_FMT}\n\n{_AGENT_EXAMPLE}\n\nQuestion: {question}")


def closed_book_instruction(question):
    return f"Answer the question below from your own knowledge. {_DIRECT_FMT}\n\nQuestion: {question}"


def rag_instruction(question):
    return ("Answer the question below. Wikipedia search results are provided between <info> and </info>. "
            f"{_DIRECT_FMT}\n\nQuestion: {question}")


def format_info(passages):
    body = "\n".join(f"Doc {i + 1} (Title: {p.get('title', '')}) {p.get('text', '')}" for i, p in enumerate(passages))
    return f"\n\n<info>\n{body or 'No results.'}\n</info>\n\n"


def extract_answer(text, lenient=False):
    m = ANSWER_RE.findall(text)
    if m:
        return m[-1].strip()
    if not lenient:
        return None
    if "<answer>" in text:  # answer opened (e.g. by the prefill) but never closed: keep its first line
        return " ".join(text.rsplit("<answer>", 1)[1].strip().split("\n")[0].split()[:12])
    # fallback for small models that skip the tags: last line outside think/info blocks, capped at 12 words
    t = re.sub(r"<info>.*?</info>|<think>.*?</think>", " ", text, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return " ".join(lines[-1].split()[:12]) if lines else ""


def run_agent(generate, retrieve, prompts, *, allow_search=True, max_searches=4, max_turn_tokens=512,
              temperature=0.0, prefill=None):
    """generate(prompts, stop, max_tokens, temperature) -> list[str]; outputs include the stop string.
    retrieve(queries) -> list[list[{title, text}]].
    prefill: optional list of {"text", "searches"} placed at the start of the assistant turn (forced RAG search).
    Returns one state dict per prompt with text, searches, answer, format_ok, n_search, stop_reason."""
    states = [{"prompt": p, "text": prefill[i]["text"] if prefill else "",
               "searches": list(prefill[i]["searches"]) if prefill else [],
               "done": False, "over_limit": 0, "stop_reason": None}
              for i, p in enumerate(prompts)]
    stop = STOP if allow_search else ["</answer>"]
    for _ in range(max_searches + 2 if allow_search else 1):
        active = [s for s in states if not s["done"]]
        if not active:
            break
        outs = generate([s["prompt"] + s["text"] for s in active], stop, max_turn_tokens, temperature)
        need, queries = [], []
        for s, out in zip(active, outs):
            s["text"] += out
            tail = out.rstrip()
            if tail.endswith("</answer>"):
                s["done"], s["stop_reason"] = True, "answer"
            elif allow_search and tail.endswith("</search>"):
                if len(s["searches"]) >= max_searches:
                    s["over_limit"] += 1
                    if s["over_limit"] >= 2:
                        s["done"], s["stop_reason"] = True, "search_limit"
                    else:
                        s["text"] += LIMIT_MSG
                else:
                    m = SEARCH_RE.findall(out)
                    need.append(s)
                    queries.append(m[-1].strip() if m else "")
            elif allow_search and tail.endswith("<info>"):
                s["done"], s["stop_reason"] = True, "fake_info"
            else:
                s["done"], s["stop_reason"] = True, "no_tag" if out else "too_long"
        if queries:
            for s, q, res in zip(need, queries, retrieve(queries)):
                s["searches"].append({"query": q, "passages": res})
                s["text"] += format_info(res)
    for s in states:
        if not s["done"]:
            s["stop_reason"] = "max_turns"
        s["answer"] = extract_answer(s["text"], lenient=True)
        s["format_ok"] = extract_answer(s["text"]) is not None
        s["n_search"] = len(s["searches"])
    return states
# conversational_memory — long-term memory across sessions

Chat history is short-lived; what the user TOLD you should not be. This
example runs an assistant across two sessions a week apart:

- **After a session**, the transcript goes through `TaguruIngester` into a
  memory context, one source id per session (`conversations/<date>`) — so
  re-memorizing a session replaces its contribution, and each session's
  facts stay attributable to the conversation they came from.
- **During a session**, every user turn first pulls relevant memories
  through `TaguruRetriever` and hands them (with provenance and weights)
  to the chat model.
- **Corrections are writes, not edits**: when the deadline moves, the new
  session's extraction asserts 締切→9月15日 at weight +1 and 締切→8月末 at
  weight -1. The outdated fact's weight cancels to 0 — history preserved,
  current answer changed.

## Run

```sh
# Python                                                    # TypeScript
cd examples/langchain                                       cd examples/langchain
.venv/bin/python conversational_memory/python/main.py       npm start --workspace=conversational_memory/typescript
```

(Setup for both is in [../README.md](../README.md). No `TAGURU_URL` → a
real server is spawned; no `OPENAI_API_KEY` → deterministic fake models.)

## What to look for

- Session 2's first turn (そば屋の提案) retrieves the そばアレルギー
  memorized a week earlier — the assistant declines for a reason it was
  never told in this session.
- After the correction, the `query` printout shows both 締切 facts with
  their weights (+1 and 0) — and the re-asked question now gets 9月15日.
- The retriever matches lexically and structurally (そば pulls そば
  facts); semantic leaps like ガレット→そば粉 belong to the chat model —
  chain a query-rewriting step in front of the retriever if you need
  those retrieved too.

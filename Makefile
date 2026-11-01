# Shortcuts for the project commands: "make chat" runs "uv run python src/scripts/chat.py".
# Run "make" or "make help" to list them.
# GNU Make is not installed on Windows by default: install it with
# "winget install ezwinports.make", or run the uv commands of the README directly.

# Question for "make ask", "make ask-sql" and "make search", e.g. make ask Q="What is a Battle-shock test?"
Q ?=
# Extra options for "make testset" and "make eval", e.g. make eval ARGS="--limit 3"
ARGS ?=

.PHONY: help install parse chunk index search ask ask-sql chat testset eval

help:
	@echo Setup:       make install, then make index
	@echo Chat:        make chat
	@echo One answer:  make ask Q=question, make ask-sql Q=question, make search Q=question
	@echo Pipeline:    make parse, make chunk, make index
	@echo Evaluation:  make testset, make eval

# Install the dependencies, including the evaluation group (RAGAS)
install:
	uv sync --group eval

# PDF -> Docling JSON + Markdown in data/processed/ (slow; the result is already in git)
parse:
	uv run python src/scripts/parse.py

# Docling JSON -> data/processed/chunks.jsonl
chunk:
	uv run python src/scripts/chunk.py

# chunks.jsonl -> embeddings in data/processed/vectors.db (needed before asking questions)
index:
	uv run python src/scripts/index.py

# Show the chunks retrieved for a question, without calling Gemini
search:
	uv run python src/scripts/search.py "$(Q)"

# One question answered from the Core Rules (PDF channel only)
ask:
	uv run python src/scripts/ask.py "$(Q)"

# One question answered from the unit database (SQL channel only)
ask-sql:
	uv run python src/scripts/ask_sql.py "$(Q)"

# Multi-turn chat with routing between the two channels
chat:
	uv run python src/scripts/chat.py

# Generate the RAGAS test set in data/eval/testset.jsonl (add ARGS="--force" to replace it)
testset:
	uv run --group eval python src/scripts/generate_testset.py $(ARGS)

# Ask the agent every test question, then score the answers with RAGAS
eval:
	uv run --group eval python src/scripts/evaluate.py $(ARGS)

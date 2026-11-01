# Project report

The [README](README.md) describes what the chatbot does, its architecture and how to run it. This report explains the key choices we made, the problems we met and how we solved them, and the evaluation results. The choices given by the course (uv, settings in `.env`, the LangGraph router, Gemini with a rate limiter, RAGAS with adapters) are not explained again.

## 1. Parsing the rulebook

We parse the PDF with Docling's standard pipeline, whose layout model finds the titles, paragraphs, tables and pictures of each page and their reading order. OCR is off because the PDF already contains its text, and we did not use the VLM pipeline, which is much slower without a GPU. We keep two outputs: the JSON, which keeps the headings and the page of every element and is the input of the chunker, and the Markdown, to read and check the result.

**Text missing from the Markdown.** Some rules are printed in decorated boxes that Docling classifies as pictures, and their text did not appear in the Markdown. Turning on OCR did not help: the problem was the Markdown export, which skips the content of pictures by default. `export_to_markdown(traverse_pictures=True)` fixed it.

**Wrong reading order on three-column pages.** While spot-checking the Markdown, we found that on p. 55 ("Muster your army") the columns are read in the wrong order: "With your opponent, select one of the battle sizes" ends up under the heading SELECT ARMY FACTION. We printed the bounding boxes of the page to understand it, but checking all 60 pages by hand was not realistic, so we left it for version 2. The text is still indexed, but some chunks carry a wrong heading.

## 2. Chunking

The course suggested LangChain's text splitters. We used Docling's HybridChunker instead because it works on the document structure: a chunk never crosses a section and keeps the headings it belongs to, whereas a character splitter cuts rules in the middle. Its size is counted with the tokenizer of our embedding model, with a maximum of 256 tokens, the input limit of `all-MiniLM-L6-v2`, so no chunk is truncated when it is embedded.

We embed the headings together with the text, because many rules only make sense with their title (e.g. "REMAIN STATIONARY: If a unit Remains Stationary, none of its models can be moved..."). The LLM receives the text alone, with the headings and the page written before it so that it can cite the page.

**Noise in the chunks.** Docling read the rulebook's bullet symbol as the control character `\x07`, which we replace with "-". Some chunks contained only a digit ("2", "3", "6", step numbers from diagrams), so we drop chunks without any letter. The result is 250 chunks (median 90 words) covering 57 of the 60 pages.

## 3. Retrieval

`all-MiniLM-L6-v2` is small, runs locally on CPU and uses none of the Gemini quota. We store the vectors with sqlite-vec, a SQLite extension: one file, no server, the same technology as the unit database. An exact search over 250 vectors is instant, so a dedicated vector database was not needed.

The model only understands English, while users may write in French or Chinese. Instead of changing the model, the agent rewrites every message as an English question before searching (section 5).

## 4. SQL channel

Gemini receives the question, with the data dictionary of the database as system prompt, and writes one SQLite query. We preferred one call with the whole dictionary to a multi-step SQL agent (list the tables, inspect them, then query), which needs more calls on the free tier.

**Traps in the data.** Looking at the database, we found several traps and wrote them as rules in the prompt. Every column is `TEXT`, so numbers compared as text give wrong results (`'10' < '4'`): the model must use `CAST`. Flags are the strings `'true'` and `'false'`. Some names contain broken apostrophes (encoding errors, e.g. T'au), so names are matched with `LIKE`. Finally, the `role` column of `datasheets` is always empty: unit types such as Infantry or Vehicle are stored as keywords.

**Running SQL written by an LLM.** The connection is read-only and only `SELECT` queries are accepted, so no query can change the database. When a query fails, Gemini receives the error and writes a corrected query, once. Results are limited to 50 rows and long texts are cut, to keep the prompt small. One unexpected bug: the `#` characters in our project folder (`##semester3`) broke the SQLite connection URI; building it with `Path.as_uri()` escapes them.

## 5. Chat agent

The graph follows the router pattern of the course. Our main design choice is the `plan` step: one Gemini call with structured output (a Pydantic schema) returns:

- the message rewritten as a standalone English question, which resolves references to earlier turns ("what is its save?") and lets the English embedding model handle any language;
- the language the answer must be written in;
- two yes/no answers: does the answer need the rulebook, and does it need the unit database? The code turns them into a route: `rules`, `database`, `both`, or `other` when neither is needed (small talk, or a request about the conversation itself, such as translating the previous answer).

Doing all this in one call keeps a message at two Gemini calls (three or four with SQL).

**Combining the two channels.** Some questions need both sources, e.g. "What is the Move of the Intercessor Squad, and how far can it move when it Advances?": the Move is in the database, the Advance rule in the book. For the route `both`, the conditional edge returns the two retrieval nodes, which LangGraph runs in parallel before a single `answer` step. Two nodes cannot write the same state field in the same step, so `contexts` and `sources` have a reducer (`operator.add`) that concatenates their lists, and the `plan` step empties them at each turn with `Overwrite([])`. The database is only needed for a specific, named unit, so that rules questions about a type of unit (Vehicle, Infantry...) stay in `rules`.

**Thread bug.** Our first version of `both` crashed with "SQLite objects created in a thread can only be used in that same thread". LangGraph runs parallel nodes in a thread pool, but a single node in the main thread, which is why the other routes never showed the problem. Each connection is only used by one node at a time, so we opened both with `check_same_thread=False`.

**Misrouted question and invented citation.** At first, the plan step chose one of four routes. It sent the Move and Advance question above to `database` only, probably because the question starts with a characteristic. The answer model then added the Advance rule from memory, with a wrong page number. We replaced the four-way choice with the two yes/no answers, so that the model judges each source separately. The answer prompt now also requires saying which part of the question the evidence does not cover, and citing only pages and units that appear in the evidence (e.g. "unit database: Intercessor Squad").

**Questions where one source decides what to look up in the other.** When we tested "Can the Intercessor Squad shoot while it is within Engagement Range?", the route was `both`, but the answer was wrong ("no"). The correct answer needs a chain: the database shows that the squad carries pistols, and only then is the Pistol rule (p. 25) relevant. Searching both sources at the same time with the same question, the SQL query looked at stratagems instead of weapons, and the vector search did not find the Pistol rule. The route `both` therefore only works when the two parts of the question are independent. A fixed order would not solve this either: some questions need the rulebook first (which exceptions exist), then the database (which units match).

**Translation bug.** In our first version, `language` meant the language of the message. When the user wrote "Traduis ta réponse en anglais", the agent detected French and answered in French. We redefined it as the language the answer must be written in, with this example in the prompt.

The answer prompt only allows the retrieved evidence, not the model's own knowledge: Warhammer rules change between editions, so an answer from memory can sound right and be wrong.

## 6. Evaluation

We generate the questions from 20 random chunks of at least 300 characters (fixed seed), to limit the number of Gemini calls. This gave 16 questions: 5 single-hop, 6 multi-hop abstract and 5 multi-hop specific. We evaluate the full chat agent, not only the retriever, so that routing and rewriting are tested too. Each answer is saved as soon as it is produced, so a run stopped by the free-tier quota resumes where it stopped. As the instructions say, the SQL channel (routes `database` and `both`) is not evaluated.

**Installation problems.** Adding RAGAS 0.4.3 failed twice: our version of `rich` was too recent for one of its dependencies (`instructor`), and RAGAS imports a module that only exists in `langchain-community` below 0.4. We pinned both versions. Our embedding adapter also crashed at first because the RAGAS base class does not set `run_config`; we set it in the constructor.

We ran the evaluation twice on the same 16 questions: before adding the route `both`, and after (route decided by two yes/no answers, stricter answer prompt).

| Metric | First run | Second run |
|---|---:|---:|
| Context precision (with reference) | 0.79 | 0.66 |
| Context recall | 0.93 | 0.93 |
| Faithfulness | 0.94 | 0.96 |
| Factual correctness (F1) | 0.71 | 0.69 |

In the second run, 15 of the 16 questions were routed to `rules`. The remaining one names the Rapid Ingress stratagem and was routed to `both`, since stratagems are also stored in the database.

Recall does not change: in both runs, the retriever finds the information needed for almost every question, and multi-hop abstract questions remain the hardest (recall 0.81). Faithfulness rises slightly, which is consistent with the new answer prompt that forbids completing an answer from memory. Factual correctness stays the lowest score, partly because this metric also penalises correct details that are not in the reference answer.

Precision drops the most, although the retriever itself did not change. Two effects explain it. The plan step rewrites each question with Gemini, and the wording varies from one run to the next: 6 of the 16 questions retrieved different chunks (one of them also received a SQL result through the route `both`). The judge is not stable either: among the 10 questions that retrieved exactly the same chunks, 2 got a different precision.

With 16 questions generated from our own chunks, and the same Gemini model as judge, differences of a few points between two runs are not meaningful. What both runs show is that the agent finds the needed information and stays faithful to it, and that the new routing keeps rulebook questions almost always in `rules`.

## 7. Limitations

Besides the problems above, the route `both` uses the same English question for the vector search and for the SQL query, instead of one sub-question per channel, and it cannot answer questions where one source decides what to look up in the other: this would need an agent that decides step by step which source to query next. The conversation memory is also lost when the program stops. The planned improvements are listed in the [Versions](README.md#versions) section of the README.

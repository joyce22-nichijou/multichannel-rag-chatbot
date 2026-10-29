import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from raglab1.llm.gemini import GeminiLLM
from raglab1.settings import settings

SQL_RULES = """You translate questions about Warhammer 40,000 units into ONE SQLite query
for the database documented below.

Rules:
- Return only the SQL query: no explanation, no markdown.
- Write a single SELECT query (WITH ... SELECT is allowed). The database is read-only.
- Every column is TEXT, numbers included. To compare numbers, use CAST, for example
  CAST(m.t AS INTEGER) >= 5. An exact value can be compared as text: m.t = '4'.
- Flags are the strings 'true' and 'false'.
- Match names with LIKE '%...%' instead of '=': the wording of the question may differ,
  and some names contain encoding errors in apostrophes (e.g. T'au, Emperor's Children).
- datasheets.role is always empty. Unit types such as Character, Infantry, Vehicle or
  Monster are stored as keywords in datasheets_keywords.keyword.
- Use SELECT DISTINCT when joins can repeat rows, and select only the columns needed.
- Add LIMIT 50 unless the question asks for a count.

Database documentation:
"""


@dataclass
class SqlResult:
    """The SQL query that was run and what it returned."""

    sql: str
    columns: list[str]
    rows: list[list]
    truncated: bool  # True when there were more rows than max_rows
    error: str | None = None


class SqlRetriever:
    """Answer data questions by letting the LLM write SQL for the unit database."""

    def __init__(
        self,
        llm: GeminiLLM,
        db_path: Path = settings.sql_db_path,
        dictionary_path: Path = settings.sql_dictionary_path,
        max_rows: int = settings.sql_max_rows,
        max_cell_length: int = settings.sql_max_cell_length,
    ) -> None:
        if not db_path.exists():
            raise FileNotFoundError(
                f"{db_path} not found: copy wahadb.sqlite into data/tables/ (it is not in git)."
            )
        self._llm = llm
        self._system_prompt = SQL_RULES + dictionary_path.read_text(encoding="utf-8")
        self._max_rows = max_rows
        self._max_cell_length = max_cell_length
        # mode=ro: the connection cannot modify the database, whatever the query is.
        # as_uri() also escapes the "#" characters of the project path.
        self._conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)

    def retrieve(self, question: str) -> SqlResult:
        """Generate a query, run it, and retry once with the error if it fails."""
        sql = self._generate_sql(question)
        try:
            return self._run(sql)
        except (sqlite3.Error, ValueError) as error:
            sql = self._generate_sql(question, failed_sql=sql, error=str(error))
            try:
                return self._run(sql)
            except (sqlite3.Error, ValueError) as error:
                return SqlResult(sql, [], [], truncated=False, error=str(error))

    def _generate_sql(
        self, question: str, failed_sql: str | None = None, error: str | None = None
    ) -> str:
        prompt = f"Question: {question}"
        if failed_sql:
            prompt += (
                f"\n\nYour previous query failed.\nQuery: {failed_sql}\n"
                f"Error: {error}\nWrite a corrected query."
            )
        return extract_sql(self._llm.generate(prompt, system=self._system_prompt))

    def _run(self, sql: str) -> SqlResult:
        if not re.match(r"\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed.")
        cursor = self._conn.execute(sql)
        columns = [description[0] for description in cursor.description]
        # Fetch one extra row to know whether the result was cut
        rows = cursor.fetchmany(self._max_rows + 1)
        return SqlResult(
            sql=sql,
            columns=columns,
            rows=[
                [self._shorten(value) for value in row]
                for row in rows[: self._max_rows]
            ],
            truncated=len(rows) > self._max_rows,
        )

    def _shorten(self, value):
        """Cut long texts (rule descriptions) so the prompt stays small."""
        if isinstance(value, str) and len(value) > self._max_cell_length:
            return value[: self._max_cell_length] + "..."
        return value


def extract_sql(text: str) -> str:
    """Remove the ```sql fences that the LLM sometimes adds around the query."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = match.group(1) if match else text
    return sql.strip().rstrip(";").strip()

from typing import ClassVar

import pytest

import main


class FakeRAG:
    instances: ClassVar[list["FakeRAG"]] = []

    def __init__(self, db_path):
        self.db_path = db_path
        self.ingested = None
        self.questions = []
        FakeRAG.instances.append(self)

    def ingest(self, documents_dir):
        self.ingested = documents_dir

    def ask(self, question):
        self.questions.append(question)
        return "The answer [S1]."


@pytest.fixture(autouse=True)
def fake_rag(monkeypatch):
    FakeRAG.instances = []
    monkeypatch.setattr(main, "RAGPipeline", FakeRAG)
    monkeypatch.setattr(main, "load_dotenv", lambda: None)


def test_ingest_flag_indexes_docs_and_exits(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py", "--ingest"])
    main.main()
    rag = FakeRAG.instances[0]
    assert rag.db_path == "./chroma_data"
    assert rag.ingested == "docs"
    assert rag.questions == []
    assert "Document ingestion completed." in capsys.readouterr().out


def test_question_is_answered(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr("builtins.input", lambda prompt: "  What is the allowance?  ")
    main.main()
    assert FakeRAG.instances[0].questions == ["What is the allowance?"]
    assert "Answer:\nThe answer [S1]." in capsys.readouterr().out


def test_blank_question_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr("builtins.input", lambda prompt: "   ")
    main.main()
    assert FakeRAG.instances[0].questions == []
    assert "Please enter a question." in capsys.readouterr().out

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_archive_ai.config import AppConfig, load_config
from local_archive_ai.services import _sanitize_text, _token_count, system_checks


class TestCore(unittest.TestCase):
    def test_load_config_uses_env_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text(
                "chunk_size: 600\n"
                "top_k: 5\n"
                "model_name: test-model\n"
                "faiss_path: data/faiss_index\n"
                "debug_mode: false\n"
                "ollama_endpoint: http://127.0.0.1:11434\n"
                "ollama_api_key: ''\n"
                "ocr_engine: tesseract\n"
                "bm25_weight: 0.4\n"
                "rerank_top_n: 5\n",
                encoding="utf-8",
            )
            original = os.environ.get("OLLAMA_API_KEY")
            try:
                os.environ["OLLAMA_API_KEY"] = "env-secret"
                loaded = load_config(Path(config_file))
                self.assertEqual(loaded.ollama_api_key, "env-secret")
                self.assertEqual(loaded.chunk_size, 600)
                self.assertEqual(loaded.top_k, 5)
                self.assertEqual(loaded.model_name, "test-model")
                self.assertEqual(loaded.ocr_engine, "tesseract")
                self.assertEqual(loaded.bm25_weight, 0.4)
                self.assertEqual(loaded.rerank_top_n, 5)
            finally:
                if original is None:
                    os.environ.pop("OLLAMA_API_KEY", None)
                else:
                    os.environ["OLLAMA_API_KEY"] = original

    def test_load_config_raises_for_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text(
                "chunk_size: not_an_int\n"
                "top_k: 5\n"
                "model_name: test-model\n"
                "faiss_path: data/faiss_index\n"
                "debug_mode: false\n"
                "ollama_endpoint: http://127.0.0.1:11434\n"
                "ocr_engine: invalid_engine\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(Path(config_file))

    def test_load_config_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text(
                "chunk_size: 500\n"
                "top_k: 4\n"
                "model_name: test-model\n"
                "faiss_path: data/faiss_index\n"
                "debug_mode: false\n"
                "ollama_endpoint: http://127.0.0.1:11434\n"
                "ocr_engine: tesseract\n"
                "retrieval_mode: vector\n"
                "unexpected_key: nope\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(Path(config_file))

    def test_sanitize_text_removes_control_chars(self) -> None:
        text = "Hello\x00World\n\nThis  is\t broken."
        sanitized = _sanitize_text(text, max_length=100)
        self.assertNotIn("\x00", sanitized)
        self.assertNotIn("\t", sanitized)
        self.assertIn("Hello World", sanitized)

    def test_token_count_returns_positive_value(self) -> None:
        count = _token_count("Hello world from the offline agent")
        self.assertGreater(count, 0)

    @patch("local_archive_ai.services.check_ollama_status", return_value=True)
    @patch("local_archive_ai.services.check_ollama_model", return_value=True)
    def test_system_checks_uses_configured_ollama_values(self, mock_model, mock_status) -> None:
        checks = system_checks(
            store_path="data/faiss_index",
            model_name="test-model",
            endpoint="http://127.0.0.1:11434",
            api_key="secret-key",
        )
        mock_status.assert_called_once_with(endpoint="http://127.0.0.1:11434", api_key="secret-key")
        mock_model.assert_called_once_with(endpoint="http://127.0.0.1:11434", api_key="secret-key", model_name="test-model")
        self.assertTrue(checks["ollama_reachable"])
        self.assertTrue(checks["ollama_model_available"])


if __name__ == "__main__":
    unittest.main()

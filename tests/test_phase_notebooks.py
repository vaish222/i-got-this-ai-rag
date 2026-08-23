from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    2: ("phase_2_baseline_evaluation.ipynb", "evaluation/run_baseline.py"),
    3: ("phase_3_chunking_experiments.ipynb", "evaluation/run_chunk_experiments.py"),
    4: ("phase_4_embedding_experiments.ipynb", "evaluation/run_embedding_experiments.py"),
    5: ("phase_5_retrieval_experiments.ipynb", "evaluation/run_retrieval_experiments.py"),
    6: ("phase_6_reranking_experiments.ipynb", "evaluation/run_reranking_experiments.py"),
    7: ("phase_7_metadata_experiments.ipynb", "evaluation/run_metadata_experiments.py"),
    8: ("phase_8_query_transformation_experiments.ipynb", "evaluation/run_query_experiments.py"),
    9: ("phase_9_langgraph_agentic_rag.ipynb", "evaluation/run_agentic_rag.py"),
    10: ("phase_10_final_evaluation.ipynb", "evaluation/run_final_evaluation.py"),
}


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


class PhaseNotebookTests(unittest.TestCase):
    def test_every_implemented_phase_has_a_notebook(self) -> None:
        phase_one = PROJECT_ROOT / "notebooks" / "phase_1_naive_dense_rag.ipynb"
        self.assertTrue(phase_one.is_file())

        for filename, _ in NOTEBOOKS.values():
            with self.subTest(filename=filename):
                self.assertTrue((PROJECT_ROOT / "notebooks" / filename).is_file())

    def test_notebooks_delegate_to_the_tested_runner_and_default_to_dry_run(self) -> None:
        for phase, (filename, runner) in NOTEBOOKS.items():
            with self.subTest(phase=phase):
                path = PROJECT_ROOT / "notebooks" / filename
                notebook = json.loads(path.read_text(encoding="utf-8"))
                source = "\n".join(cell_source(cell) for cell in notebook["cells"])

                self.assertEqual(notebook["nbformat"], 4)
                self.assertEqual(notebook["metadata"]["i_got_this_rag"]["phase"], phase)
                self.assertEqual(
                    notebook["metadata"]["i_got_this_rag"]["source_runner"],
                    runner,
                )
                self.assertIn(Path(runner).name, source)
                self.assertIn("RUN_EXPERIMENT = False", source)
                self.assertNotIn("import streamlit", source.lower())

    def test_all_new_notebook_code_cells_parse_and_have_no_saved_outputs(self) -> None:
        for phase, (filename, _) in NOTEBOOKS.items():
            path = PROJECT_ROOT / "notebooks" / filename
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cell_ids: set[str] = set()
            for index, cell in enumerate(notebook["cells"], start=1):
                with self.subTest(phase=phase, cell=index, field="id"):
                    self.assertTrue(cell.get("id"))
                    self.assertNotIn(cell["id"], cell_ids)
                    cell_ids.add(cell["id"])
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(phase=phase, cell=index):
                    ast.parse(cell_source(cell), filename=f"{path}:cell-{index}")
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()

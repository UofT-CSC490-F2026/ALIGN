import contextlib
import io
import json
import runpy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.coverage_summary import END, START, main as write_coverage
from test_process_runs import FIXTURE, process_runs, run_artifacts
import evaluate


class CoverageReportingTests(unittest.TestCase):
    def test_readme_refresh_is_idempotent_and_preserves_other_content(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report, readme = root / "report.json", root / "README.md"
            report.write_text(json.dumps({"totals":{"covered_lines":8, "num_statements":10,
                                                    "covered_branches":3, "num_branches":4}}), encoding="utf-8")
            readme.write_text(f"# Title\n{START}\nold\n{END}\nfooter\n", encoding="utf-8")
            write_coverage([str(report), "--readme", str(readme)])
            updated = readme.read_text(encoding="utf-8")
            self.assertIn("| Executable lines | 8 | 10 | 80.00% |", updated)
            self.assertIn("| Branches | 3 | 4 | 75.00% |", updated)
            self.assertTrue(updated.startswith("# Title\n"))
            self.assertTrue(updated.endswith("footer\n"))
            write_coverage([str(report), "--readme", str(readme)])
            self.assertEqual(readme.read_text(encoding="utf-8"), updated)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                write_coverage([str(report)])
            self.assertIn("75.00%", output.getvalue())


class A2CliTests(unittest.TestCase):
    def invoke(self, module, *args):
        with patch.object(sys, "argv", [module.__file__, *args]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            module.main()
        return output.getvalue()

    def test_process_and_evaluation_dry_runs_and_apply_without_credentials(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            run_artifacts.stage(FIXTURE, bundle)
            for module, args in ((process_runs, ["--run-dir", str(bundle)]),
                                 (evaluate, ["--run-key", "fixture"])):
                with self.subTest(module=module.__name__), patch.dict("os.environ", {}, clear=True):
                    self.assertIn("DRY RUN", self.invoke(module, *args))
                    with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
                        self.invoke(module, *args, "--apply")
                    self.assertIn("requires ALIGN_DATABASE_URL", error.getvalue())

    def test_a2_script_help(self):
        for module in (process_runs, evaluate):
            with self.subTest(module=module.__name__), patch.object(sys, "argv", [module.__file__, "--help"]), \
                 contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as exit:
                runpy.run_path(module.__file__, run_name="__main__")
            self.assertEqual(exit.exception.code, 0)
            self.assertIn("--apply", output.getvalue())

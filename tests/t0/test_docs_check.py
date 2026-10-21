"""tools/docs_check.py: cases on a throwaway tree, plus the docs of the real repository."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import docs_check  # noqa: E402


def tree(tmp_path, files):
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_clean_tree_has_no_problems(tmp_path):
    tree(tmp_path, {"README.md": "# Title\n\nSee [guide](docs/a.md#second-part) and [top](#title).\n",
                    "docs/a.md": "# A\n\n## Second part\n\ntext\n"})
    assert docs_check.check(tmp_path) == []


def test_broken_link_and_anchor(tmp_path):
    tree(tmp_path, {"README.md": "# T\n\n[a](docs/missing.md) [b](docs/a.md#nope) [c](https://example.com/x)\n",
                    "docs/a.md": "# A\n"})
    found = docs_check.check(tmp_path)
    assert any("broken link docs/missing.md" in p for p in found)
    assert any("no heading for anchor docs/a.md#nope" in p for p in found)
    assert len(found) == 2


def test_duplicate_headings_get_numbered_anchors(tmp_path):
    tree(tmp_path, {"README.md": "# T\n\n[x](docs/a.md#notes-1)\n", "docs/a.md": "# A\n\n## Notes\n\n## Notes\n"})
    assert docs_check.check(tmp_path) == []


def test_forbidden_characters_outside_fences_only(tmp_path):
    tree(tmp_path, {"README.md": "# T\n\nprose — dash and a → arrow\n\n```\nVerdict: PASS — ok\n```\n"})
    found = docs_check.check(tmp_path)
    assert sorted(p.split(": ", 1)[1] for p in found) == ["arrow", "em dash"]
    assert all(p.startswith("README.md:3:") for p in found)


def test_local_paths_and_ssh_are_flagged_even_in_fences(tmp_path):
    tree(tmp_path, {"docs/a.md": "# A\n\n```\ncd /ho" "me/someone/project\nss" "h bob@example\n```\n/Us" "ers/me/x\n"})
    found = docs_check.check(tmp_path)
    assert sorted(p.split(": ", 1)[1] for p in found) == ["local path", "local path", "ssh login"]


def test_placeholders_are_not_local_paths(tmp_path):
    tree(tmp_path, {"docs/a.md": "# A\n\nUse `/home/<user>/slotstream-spark` or `/Users/$USER/x`.\n"})
    assert docs_check.check(tmp_path) == []


def test_terms_file(tmp_path):
    tree(tmp_path, {"docs/a.md": "# A\n\nthe Secret-Host box\n"})
    terms = tmp_path / "terms.txt"
    terms.write_text("# comment\n\nsecret-host\n", encoding="utf-8")
    found = docs_check.check(tmp_path, docs_check.load_terms(terms))
    assert found == ["docs/a.md:3: private term"]


def test_main_prints_count_and_exit_code(tmp_path, capsys):
    tree(tmp_path, {"README.md": "# T\n\n[a](nope.md)\n"})
    assert docs_check.main(["--root", str(tmp_path)]) == 1
    assert capsys.readouterr().out.strip().endswith("docs check: 1 problems")


def test_docs_folder_of_the_repository_is_clean():
    found = [p for p in docs_check.check(ROOT) if p.startswith(("docs/", "db/DB.md"))]
    assert found == []

"""Tests for the company profile and knowledge search (shared/company.py).

conftest.py selects the made-up sample company (companies/example); tests that
need something else build a company folder in tmp_path.
"""

import os
import textwrap
import time

import pytest

from shared import company


def _company(tmp_path, profile: str, docs: dict[str, str] | None = None):
    root = tmp_path / "acme"
    (root / "knowledge").mkdir(parents=True)
    (root / "profile.yaml").write_text(textwrap.dedent(profile), encoding="utf-8")
    for name, text in (docs or {}).items():
        path = root / "knowledge" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return root


PROFILE = """\
name: Acme
summary: Makes anvils.
offer: ["Anvils", "Rockets"]
rules: ["Never promise delivery dates."]
"""


@pytest.fixture
def acme(tmp_path, monkeypatch):
    root = _company(tmp_path, PROFILE, {
        "pricing.md": """\
            # Pricing
            ## Anvils
            A standard anvil costs $120. Delivery is $10.
            ## Rockets
            Rockets cost $900 each and are returned within 30 days.
            """,
        "faq/returns.txt": "Returns are accepted within 14 days of delivery.\n",
        "ignored.pdf": "not indexed",
    })
    monkeypatch.setenv("ERGON_COMPANY", str(root))
    return root


# --------------------------------------------------------------------------- #
# Which company
# --------------------------------------------------------------------------- #
class TestCompanyDir:
    def test_default_is_the_sample_company(self, monkeypatch):
        monkeypatch.delenv("ERGON_COMPANY", raising=False)
        assert company.company_dir() == company.COMPANIES_DIR / "example"

    def test_bare_id_resolves_under_companies(self, monkeypatch):
        monkeypatch.setenv("ERGON_COMPANY", "example")
        assert company.company_dir().name == "example"

    def test_path_is_used_as_is(self, acme):
        assert company.company_dir() == acme

    def test_unknown_company_lists_the_known_ones(self, monkeypatch):
        monkeypatch.setenv("ERGON_COMPANY", "nobody")
        with pytest.raises(company.CompanyConfigError, match="example"):
            company.company_dir()

    def test_every_shipped_company_loads(self, monkeypatch):
        for path in company.COMPANIES_DIR.iterdir():
            if (path / "profile.yaml").is_file():
                monkeypatch.setenv("ERGON_COMPANY", path.name)
                assert company.load_profile().name
                assert company.knowledge_index().sections


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
class TestProfile:
    def test_load_and_render(self, acme):
        profile = company.load_profile()
        assert profile.name == "Acme"
        assert profile.offer == ["Anvils", "Rockets"]
        text = company.render_profile(profile)
        assert text.startswith("## Your company: Acme")
        assert "- Rockets" in text
        assert "Never promise delivery dates." in text

    @pytest.mark.parametrize("missing", ["name", "summary"])
    def test_required_fields(self, tmp_path, monkeypatch, missing):
        text = "\n".join(ln for ln in PROFILE.splitlines() if not ln.startswith(missing))
        monkeypatch.setenv("ERGON_COMPANY", str(_company(tmp_path, text)))
        with pytest.raises(company.CompanyConfigError, match=missing):
            company.load_profile()

    def test_rules_must_be_a_list(self, tmp_path, monkeypatch):
        root = _company(tmp_path, PROFILE.replace('rules: ["Never promise delivery dates."]',
                                                  "rules: be nice"))
        monkeypatch.setenv("ERGON_COMPANY", str(root))
        with pytest.raises(company.CompanyConfigError, match="rules"):
            company.load_profile()

    def test_edits_are_picked_up_without_restart(self, acme):
        assert company.load_profile().summary == "Makes anvils."
        path = acme / "profile.yaml"
        path.write_text(PROFILE.replace("Makes anvils.", "Makes rockets."), encoding="utf-8")
        later = time.time() + 5
        os.utime(path, (later, later))
        assert company.load_profile().summary == "Makes rockets."


# --------------------------------------------------------------------------- #
# Knowledge
# --------------------------------------------------------------------------- #
class TestSections:
    def test_heading_path_is_kept(self):
        sections = company.split_sections("p.md", "# Pricing\nintro\n## Anvils\n$120\n### Big\n$200\n")
        assert [(s.heading, s.text) for s in sections] == [
            ("Pricing", "intro"),
            ("Pricing > Anvils", "$120"),
            ("Pricing > Anvils > Big", "$200"),
        ]

    def test_text_without_headings_uses_the_file_name(self):
        [section] = company.split_sections("notes.txt", "just text")
        assert section.heading == "notes.txt"

    def test_long_section_is_split_on_paragraphs(self):
        para = "word " * 200
        sections = company.split_sections("x.md", f"# T\n{para}\n\n{para}\n\n{para}\n")
        assert len(sections) > 1
        assert all(s.heading == "T" for s in sections)


class TestSearch:
    async def test_best_section_first(self, acme):
        out = await company.search_company_knowledge("rocket price")
        assert out["success"] is True
        assert out["company"] == "Acme"
        assert out["results"][0]["section"] == "Pricing > Rockets"

    async def test_stemming_matches_word_forms(self, acme):
        out = await company.search_company_knowledge("returning a rocket")
        sections = [r["section"] for r in out["results"]]
        assert "Pricing > Rockets" in sections
        assert "faq/returns.txt" in sections

    async def test_only_markdown_and_text_are_indexed(self, acme):
        out = await company.search_company_knowledge("indexed")
        assert out["count"] == 0
        assert "Nothing matched" in out["note"]

    async def test_empty_query(self, acme):
        assert (await company.search_company_knowledge("  "))["success"] is False

    async def test_max_results_is_clamped(self, acme):
        out = await company.search_company_knowledge("anvil rocket returns", max_results=99)
        assert out["count"] <= 10

    async def test_new_file_is_indexed_without_restart(self, acme):
        assert (await company.search_company_knowledge("hammers"))["count"] == 0
        (acme / "knowledge" / "hammers.md").write_text("# Hammers\nHammers cost $15.\n",
                                                       encoding="utf-8")
        out = await company.search_company_knowledge("hammers")
        assert out["results"][0]["document"] == "hammers.md"

    async def test_bad_company_is_an_error_not_an_exception(self, monkeypatch):
        monkeypatch.setenv("ERGON_COMPANY", "nobody")
        out = await company.search_company_knowledge("anything")
        assert out["success"] is False

    async def test_non_latin_text_is_searchable(self, tmp_path, monkeypatch):
        root = _company(tmp_path, PROFILE, {"ar.md": "# الأسعار\nسعر السندان 120 دولار\n"})
        monkeypatch.setenv("ERGON_COMPANY", str(root))
        out = await company.search_company_knowledge("سعر السندان")
        assert out["results"][0]["document"] == "ar.md"


class TestReadDocument:
    async def test_list_documents(self, acme):
        out = await company.read_company_document()
        assert out == {"success": True, "documents": ["faq/returns.txt", "pricing.md"]}

    async def test_read_one(self, acme):
        out = await company.read_company_document("pricing.md")
        assert out["success"] is True
        assert "$900" in out["text"]
        assert out["truncated"] is False

    async def test_unknown_document(self, acme):
        out = await company.read_company_document("../profile.yaml")
        assert out["success"] is False
        assert out["documents"] == ["faq/returns.txt", "pricing.md"]


# --------------------------------------------------------------------------- #
# Instructions
# --------------------------------------------------------------------------- #
class TestCompanyBlock:
    def test_employee_with_knowledge_tools_is_told_to_use_them(self, acme):
        text = company.company_block({"search_company_knowledge"})
        assert "## Your company: Acme" in text
        assert "search_company_knowledge" in text
        assert "pricing.md" in text

    def test_employee_without_knowledge_tools_gets_the_profile_only(self, acme):
        text = company.company_block({"web_search"})
        assert "## Your company: Acme" in text
        assert "search_company_knowledge" not in text

    def test_chief_is_told_to_delegate(self, acme):
        assert "delegate" in company.company_block(for_chief=True)

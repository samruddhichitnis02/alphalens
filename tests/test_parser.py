import pytest
from src.edgar.parser import EdgarParser

SAMPLE_10K_HTML = """
<html>
<body>
<h1>ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)</h1>
<p>Table of Contents</p>
<p>Item 1. Business ...... 4</p>
<p>Item 1A. Risk Factors ...... 12</p>

<h2>Item 1. Business</h2>
<p>Apple designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories.</p>

<h2>Item 1A. Risk Factors</h2>
<p>Our business, reputation, results of operations and financial condition could be materially adversely affected by global macroeconomic volatility and supply chain disruptions.</p>

<h2>Item 7. Management's Discussion and Analysis</h2>
<p>Total net sales increased 8% driven by growth in Services and iPhone sales across international markets.</p>

<h2>Item 8. Financial Statements</h2>
<p>Consolidated Statements of Operations and Comprehensive Income.</p>
</body>
</html>
"""

SAMPLE_8K_HTML = """
<html>
<body>
<h1>CURRENT REPORT PURSUANT TO SECTION 13 OR 15(d)</h1>
<h2>Item 2.02. Results of Operations and Financial Condition</h2>
<p>On October 25, the Registrant issued a press release announcing its financial results for the fourth quarter.</p>

<h2>Item 5.02. Departure of Directors or Certain Officers; Election of Directors</h2>
<p>Effective November 1, the Board of Directors appointed a new Chief Financial Officer.</p>
</body>
</html>
"""

def test_html_to_clean_text():
    parser = EdgarParser()
    text = parser.html_to_text(SAMPLE_10K_HTML)
    assert "Apple designs, manufactures" in text
    assert "<html>" not in text
    assert "<p>" not in text

def test_10k_section_detection():
    parser = EdgarParser()
    text = parser.html_to_text(SAMPLE_10K_HTML)
    sections = parser.parse_filing_sections(text, "10-K")

    section_names = [s["section_name"] for s in sections]
    assert any("Item 1A - Risk Factors" in name for name in section_names)
    assert any("Item 7 - MD&A" in name for name in section_names)
    assert any("Item 8 - Financial Statements" in name for name in section_names)

    # Verify character offsets
    for s in sections:
        assert s["char_start"] < s["char_end"]
        extracted_text = text[s["char_start"]:s["char_end"]].strip()
        assert extracted_text == s["raw_text"].strip()

def test_8k_section_detection():
    parser = EdgarParser()
    text = parser.html_to_text(SAMPLE_8K_HTML)
    sections = parser.parse_filing_sections(text, "8-K")

    section_names = [s["section_name"] for s in sections]
    assert any("Item 2.02" in name for name in section_names)
    assert any("Item 5.02" in name for name in section_names)

    # Check 2.02 contents
    item_202 = next(s for s in sections if "Item 2.02" in s["section_name"])
    assert "financial results for the fourth quarter" in item_202["raw_text"]


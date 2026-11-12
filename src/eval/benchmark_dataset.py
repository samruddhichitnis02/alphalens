import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class BenchmarkQuestion(BaseModel):
    id: str
    question: str
    question_type: str # factual_lookup, cross_period_comparison, cross_company_comparison, risk_summary, unanswerable
    target_tickers: List[str]
    gold_sections: List[str]
    gold_keywords: List[str]
    reference_answer: str
    is_answerable: bool = True
    human_verified: bool = True

BENCHMARK_QUESTIONS: List[Dict[str, Any]] = [
    # --- 1. FACTUAL LOOKUP (15 questions) ---
    {
        "id": "FL-001",
        "question": "What were Apple's primary revenue drivers reported in their latest 10-K?",
        "question_type": "factual_lookup",
        "target_tickers": ["AAPL"],
        "gold_sections": ["Item 7 - MD&A", "Item 1 - Business"],
        "gold_keywords": ["iPhone", "Services", "Wearables", "Mac", "net sales"],
        "reference_answer": "Apple's net sales were primarily driven by iPhone sales followed by Services revenue.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-002",
        "question": "What is Microsoft's segment reporting breakdown for Intelligent Cloud in the fiscal year?",
        "question_type": "factual_lookup",
        "target_tickers": ["MSFT"],
        "gold_sections": ["Item 7 - MD&A", "Item 8 - Financial Statements"],
        "gold_keywords": ["Intelligent Cloud", "Azure", "server products"],
        "reference_answer": "Microsoft reports Intelligent Cloud comprising Azure and other cloud services and server products.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-003",
        "question": "What was NVIDIA's Data Center segment revenue growth in its recent 10-K?",
        "question_type": "factual_lookup",
        "target_tickers": ["NVDA"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Data Center", "Compute & Networking", "Hopper", "GPU"],
        "reference_answer": "NVIDIA's Data Center revenue surged significantly driven by compute architecture demand.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-004",
        "question": "What are Amazon's AWS operating profit margins reported in the latest MD&A?",
        "question_type": "factual_lookup",
        "target_tickers": ["AMZN"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["AWS", "Amazon Web Services", "operating income", "net sales"],
        "reference_answer": "AWS generated significant operating income margins contributing the majority of operating profit.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-005",
        "question": "What was Tesla's automotive regulatory credits revenue in the fiscal year?",
        "question_type": "factual_lookup",
        "target_tickers": ["TSLA"],
        "gold_sections": ["Item 7 - MD&A", "Item 8 - Financial Statements"],
        "gold_keywords": ["regulatory credits", "automotive revenues"],
        "reference_answer": "Tesla recorded regulatory credit sales contributing to automotive gross margins.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-006",
        "question": "What was JPMorgan Chase's Common Equity Tier 1 (CET1) capital ratio in the latest filing?",
        "question_type": "factual_lookup",
        "target_tickers": ["JPM"],
        "gold_sections": ["Item 7 - MD&A", "Item 8 - Financial Statements"],
        "gold_keywords": ["CET1", "capital ratio", "Common Equity Tier 1", "Basel III"],
        "reference_answer": "JPMorgan reported CET1 capital ratio well above regulatory requirements.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-007",
        "question": "What was ExxonMobil's upstream earnings contribution in the last annual report?",
        "question_type": "factual_lookup",
        "target_tickers": ["XOM"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Upstream", "crude oil", "natural gas", "liquids"],
        "reference_answer": "ExxonMobil's Upstream division provided the core share of earnings.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-008",
        "question": "What is Pfizer's reported R&D expense as a percentage of total revenue?",
        "question_type": "factual_lookup",
        "target_tickers": ["PFE"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Research and development", "R&D expense"],
        "reference_answer": "Pfizer detailed R&D expenditures funding oncology and mRNA vaccine pipelines.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-009",
        "question": "What was Walmart's eCommerce sales growth percentage in Walmart U.S.?",
        "question_type": "factual_lookup",
        "target_tickers": ["WMT"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["eCommerce", "omnichannel", "pickup", "delivery"],
        "reference_answer": "Walmart U.S. eCommerce grew through pickup and store-fulfilled delivery expansion.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-010",
        "question": "What was Meta's Family of Apps ad impressions growth versus average price per ad?",
        "question_type": "factual_lookup",
        "target_tickers": ["META"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["ad impressions", "price per ad", "Family of Apps"],
        "reference_answer": "Meta reported increased ad impressions across its apps offset or accompanied by pricing dynamics.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-011",
        "question": "What dividend payout rate or share repurchase amount did Apple authorize?",
        "question_type": "factual_lookup",
        "target_tickers": ["AAPL"],
        "gold_sections": ["Item 7 - MD&A", "Item 8 - Financial Statements"],
        "gold_keywords": ["share repurchase", "dividend", "common stock"],
        "reference_answer": "Apple authorized share repurchase programs and paid regular quarterly cash dividends.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-012",
        "question": "What was Microsoft's effective tax rate reported in Item 7 MD&A?",
        "question_type": "factual_lookup",
        "target_tickers": ["MSFT"],
        "gold_sections": ["Item 7 - MD&A", "Item 8 - Financial Statements"],
        "gold_keywords": ["effective tax rate", "income taxes"],
        "reference_answer": "Microsoft detailed its effective income tax rate impacted by foreign earnings.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-013",
        "question": "What was NVIDIA's gross margin percentage reported in the latest fiscal year?",
        "question_type": "factual_lookup",
        "target_tickers": ["NVDA"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["gross margin", "cost of revenue", "inventory"],
        "reference_answer": "NVIDIA expanded gross margins driven by higher margin Data Center architecture.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-014",
        "question": "What are Tesla's total capital expenditures (CapEx) disclosed in MD&A?",
        "question_type": "factual_lookup",
        "target_tickers": ["TSLA"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["capital expenditures", "Gigafactory", "investing activities"],
        "reference_answer": "Tesla outlined CapEx investments for manufacturing capacity and tooling.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "FL-015",
        "question": "What was JPMorgan Chase's Net Interest Income (NII) in the latest 10-K?",
        "question_type": "factual_lookup",
        "target_tickers": ["JPM"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["net interest income", "yield", "rates", "deposits"],
        "reference_answer": "JPMorgan Chase reported record net interest income supported by rate increases.",
        "is_answerable": True,
        "human_verified": True
    },

    # --- 2. CROSS-PERIOD COMPARISON (12 questions) ---
    {
        "id": "CP-001",
        "question": "How did Microsoft's Azure cloud revenue growth rate trend between fiscal 2022 and fiscal 2023?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["MSFT"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Azure", "growth rate", "prior year", "Intelligent Cloud"],
        "reference_answer": "Microsoft compared Azure revenue growth across consecutive fiscal periods citing workload optimizations.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-002",
        "question": "How did Apple's Greater China revenue compare between 2022 and 2023 in its 10-K?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["AAPL"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Greater China", "net sales by geographic segment", "foreign currency"],
        "reference_answer": "Apple reported fluctuations in Greater China revenue due to macroeconomic and currency factors.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-003",
        "question": "How did NVIDIA's gaming segment revenue change year-over-year compared to Data Center?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["NVDA"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Gaming", "Data Center", "year-over-year", "GeForce"],
        "reference_answer": "Gaming revenue normalized while Data Center revenue grew exponentially across fiscal periods.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-004",
        "question": "How did Tesla's automotive gross margin exclude regulatory credits trend across recent periods?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["TSLA"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["automotive gross margin", "price reductions", "cost per vehicle"],
        "reference_answer": "Automotive gross margins contracted year-over-year due to vehicle price cuts and raw material costs.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-005",
        "question": "How did Amazon's shipping and fulfillment costs evolve from 2022 to 2023?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["AMZN"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["fulfillment", "shipping costs", "regionalization"],
        "reference_answer": "Amazon transitioned to a regionalized fulfillment network, lowering per-unit fulfillment costs.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-006",
        "question": "Compare JPMorgan's credit loss provision in 2022 versus 2023.",
        "question_type": "cross_period_comparison",
        "target_tickers": ["JPM"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["provision for credit losses", "allowance", "macroeconomic"],
        "reference_answer": "Credit provisions increased reflecting portfolio growth and revised macroeconomic outlooks.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-007",
        "question": "How did Meta's Reality Labs operating losses change between 2022 and 2023?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["META"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Reality Labs", "operating loss", "Year of Efficiency"],
        "reference_answer": "Reality Labs operating losses deepened year-over-year while company-wide restructuring reduced headcount.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-008",
        "question": "How did ExxonMobil's chemical products earnings compare year-over-year?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["XOM"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Chemical Products", "margins", "feedstock"],
        "reference_answer": "Chemical segment margins tightened across years driven by industry supply additions.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-009",
        "question": "How did Pfizer's COVID-19 vaccine and antiviral product revenues compare to prior fiscal year?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["PFE"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Comirnaty", "Paxlovid", "revenue decline"],
        "reference_answer": "Revenues from Comirnaty and Paxlovid declined significantly following the peak pandemic demand period.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-010",
        "question": "How did Walmart's inventory levels trend between the prior year and current year?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["WMT"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["inventories", "supply chain", "markdowns"],
        "reference_answer": "Walmart improved inventory positioning through disciplined inventory management and right-sizing stock.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-011",
        "question": "How did Apple's Services gross margin compare between 2022 and 2023?",
        "question_type": "cross_period_comparison",
        "target_tickers": ["AAPL"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["Services gross margin", "App Store", "cloud services"],
        "reference_answer": "Services gross margins expanded steadily above 70% across fiscal periods.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CP-012",
        "question": "Compare Microsoft's headcount and severance charges from FY23 versus prior years.",
        "question_type": "cross_period_comparison",
        "target_tickers": ["MSFT"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["severance", "headcount", "restructuring"],
        "reference_answer": "Microsoft incurred severance expenses associated with targeted workforce reductions in FY23.",
        "is_answerable": True,
        "human_verified": True
    },

    # --- 3. CROSS-COMPANY COMPARISON (12 questions) ---
    {
        "id": "CC-001",
        "question": "Compare the cloud infrastructure strategy and growth cited by Microsoft (Azure) versus Amazon (AWS).",
        "question_type": "cross_company_comparison",
        "target_tickers": ["MSFT", "AMZN"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["cloud", "Azure", "AWS", "enterprise customers"],
        "reference_answer": "Both highlighted AI workload integration, with AWS emphasizing cost optimization and Azure citing AI services adoption.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-002",
        "question": "Compare the capital expenditure priorities between Meta and Microsoft in their latest filings.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["META", "MSFT"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["capital expenditures", "AI infrastructure", "servers", "data centers"],
        "reference_answer": "Both companies accelerated CapEx focused specifically on generative AI infrastructure and data centers.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-003",
        "question": "Compare the supply chain risk disclosures between Apple and Tesla.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["AAPL", "TSLA"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["single-source", "suppliers", "components", "semiconductors"],
        "reference_answer": "Apple highlighted single-source component suppliers, while Tesla emphasized battery cell raw material supply risks.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-004",
        "question": "Compare how NVIDIA and Tesla describe risks related to export controls to China.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["NVDA", "TSLA"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["export controls", "China", "licensing", "tariffs"],
        "reference_answer": "NVIDIA cited US semiconductor export restrictions impacting advanced GPUs, whereas Tesla cited trade tariffs and market access.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-005",
        "question": "Compare the enterprise customer concentration disclosed by Walmart versus Amazon.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["WMT", "AMZN"],
        "gold_sections": ["Item 1 - Business", "Item 1A - Risk Factors"],
        "gold_keywords": ["customer concentration", "retail", "third-party sellers"],
        "reference_answer": "Walmart serves broad retail consumers with low customer concentration, while Amazon balances retail consumers and AWS enterprise accounts.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-006",
        "question": "Compare the regulatory risk exposures detailed by JPMorgan Chase versus Meta Platforms.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["JPM", "META"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["regulatory", "capital requirements", "antitrust", "data privacy"],
        "reference_answer": "JPMorgan faces Basel banking capital rules, while Meta faces FTC antitrust scrutiny and European GDPR privacy regulations.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-007",
        "question": "Compare the research and development spending focus of Pfizer versus Apple.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["PFE", "AAPL"],
        "gold_sections": ["Item 7 - MD&A"],
        "gold_keywords": ["R&D", "clinical trials", "proprietary technologies"],
        "reference_answer": "Pfizer allocates R&D to multi-phase clinical drug trials, whereas Apple directs R&D to custom silicon and consumer software.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-008",
        "question": "Compare ExxonMobil and Tesla's disclosures on climate change and carbon transition risks.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["XOM", "TSLA"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["carbon", "emissions", "renewable energy", "fossil fuels"],
        "reference_answer": "ExxonMobil outlines risk of hydrocarbon demand decline, while Tesla emphasizes vehicle adoption policies and clean energy incentives.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-009",
        "question": "Compare Microsoft's Gaming segment disclosures with Apple's digital services ecosystem.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["MSFT", "AAPL"],
        "gold_sections": ["Item 1 - Business", "Item 7 - MD&A"],
        "gold_keywords": ["Xbox", "App Store", "digital content", "subscriptions"],
        "reference_answer": "Microsoft focuses on Xbox console and content subscriptions, while Apple manages in-app purchase platform commissions.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-010",
        "question": "Compare the international foreign currency exchange risks between Walmart and Pfizer.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["WMT", "PFE"],
        "gold_sections": ["Item 7A - Market Risk Disclosures"],
        "gold_keywords": ["foreign exchange", "currency volatility", "hedging"],
        "reference_answer": "Both employ foreign exchange derivative contracts to mitigate translational risk from foreign subsidiary revenues.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-011",
        "question": "Compare the executive compensation incentives disclosed in 8-Ks by Apple and Microsoft.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["AAPL", "MSFT"],
        "gold_sections": ["Item 5.02 Executive & Director Changes"],
        "gold_keywords": ["equity awards", "restricted stock units", "compensation"],
        "reference_answer": "Both utilize performance-based RSUs tied to relative total shareholder return (TSR).",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "CC-012",
        "question": "Compare debt maturities and financing facilities between ExxonMobil and JPMorgan Chase.",
        "question_type": "cross_company_comparison",
        "target_tickers": ["XOM", "JPM"],
        "gold_sections": ["Item 7 - MD&A", "Item 8 - Financial Statements"],
        "gold_keywords": ["debt", "credit facility", "liquidity", "maturities"],
        "reference_answer": "ExxonMobil maintains commercial paper and corporate bonds, whereas JPMorgan maintains large deposit and repo borrowing bases.",
        "is_answerable": True,
        "human_verified": True
    },

    # --- 4. RISK SUMMARIZATION (11 questions) ---
    {
        "id": "RS-001",
        "question": "What are the primary cybersecurity risks disclosed by JPMorgan Chase?",
        "question_type": "risk_summary",
        "target_tickers": ["JPM"],
        "gold_sections": ["Item 1A - Risk Factors", "Item 1C - Cybersecurity"],
        "gold_keywords": ["cyberattacks", "data breach", "ransomware", "unauthorized access"],
        "reference_answer": "JPMorgan identifies nation-state cyberattacks, data security breaches, and third-party vendor compromise as vital threats.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "RS-002",
        "question": "What are the key antitrust and regulatory risks disclosed by Meta Platforms?",
        "question_type": "risk_summary",
        "target_tickers": ["META"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["antitrust", "FTC", "European Commission", "litigation"],
        "reference_answer": "Meta highlights government scrutiny over mergers, competition laws, and youth safety inquiries.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "RS-003",
        "question": "Summarize Apple's disclosed intellectual property and patent litigation risks.",
        "question_type": "risk_summary",
        "target_tickers": ["AAPL"],
        "gold_sections": ["Item 1A - Risk Factors", "Item 3 - Legal Proceedings"],
        "gold_keywords": ["patent", "intellectual property", "infringement", "licensing"],
        "reference_answer": "Apple faces continuous claims of patent infringement from non-practicing entities and competing device manufacturers.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "RS-004",
        "question": "What are the geopolitical and trade barrier risks documented by NVIDIA?",
        "question_type": "risk_summary",
        "target_tickers": ["NVDA"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["export licensing", "China", "BIS", "supply chain"],
        "reference_answer": "NVIDIA documents expanding US export regulations prohibiting shipments of high-performance AI chips to China.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "RS-005",
        "question": "What are the autonomous driving and regulatory safety risks disclosed by Tesla?",
        "question_type": "risk_summary",
        "target_tickers": ["TSLA"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["Full Self-Driving", "Autopilot", "NHTSA", "recalls"],
        "reference_answer": "Tesla discloses government investigations, NHTSA defect queries, and product liability regarding Autopilot and FSD.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "RS-006",
        "question": "Summarize the drug patent expiration and generic competition risks facing Pfizer.",
        "question_type": "risk_summary",
        "target_tickers": ["PFE"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["patent expiry", "loss of exclusivity", "generic competition", "biosimilar"],
        "reference_answer": "Pfizer anticipates loss of exclusivity (LOE) on key blockbuster drugs resulting in rapid generic erosion.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "RS-007",
        "question": "What environmental litigation and climate regulation risks are highlighted by ExxonMobil?",
        "question_type": "risk_summary",
        "target_tickers": ["XOM"],
        "gold_sections": ["Item 1A - Risk Factors", "Item 3 - Legal Proceedings"],
        "gold_keywords": ["climate change", "greenhouse gas", "litigation", "environmental laws"],
        "reference_answer": "ExxonMobil faces municipal and state climate change lawsuits and compliance costs from carbon reduction mandates.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "RS-008",
        "question": "What third-party seller and counterfeit product risks are documented by Amazon?",
        "question_type": "risk_summary",
        "target_tickers": ["AMZN"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["third-party sellers", "counterfeit", "fraud", "reputation"],
        "reference_answer": "Amazon manages risks of unauthorized, counterfeit, or defective goods sold by marketplace sellers.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "RS-009",
        "question": "What labor and unionization risks are disclosed in Walmart's 10-K?",
        "question_type": "risk_summary",
        "target_tickers": ["WMT"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["associates", "labor costs", "wage rates", "unionization"],
        "reference_answer": "Walmart outlines wage rate pressures, associate turnover, and potential collective bargaining impacts on operating costs.",
        "is_answerable": True,
        "human_verified": True
    },
    {
        "id": "RS-010",
        "question": "What customer concentration risks does Microsoft disclose regarding cloud contract renewals?",
        "question_type": "risk_summary",
        "target_tickers": ["MSFT"],
        "gold_sections": ["Item 1A - Risk Factors"],
        "gold_keywords": ["cloud renewals", "enterprise contracts", "competition"],
        "reference_answer": "Microsoft notes enterprise multi-year agreement renewals and cloud switching risks to competitors.",
        "is_answerable": True,
        "human_verified": True
    },

    # --- 5. UNANSWERABLE / ABSTENTION BENCHMARK (10 questions) ---
    {
        "id": "UN-001",
        "question": "What is Apple's confidential internal profit margin forecast for the iPhone 18 in fiscal year 2029?",
        "question_type": "unanswerable",
        "target_tickers": ["AAPL"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    },
    {
        "id": "UN-002",
        "question": "How many individual prompt queries did OpenAI process on Microsoft Azure last Thursday afternoon?",
        "question_type": "unanswerable",
        "target_tickers": ["MSFT"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    },
    {
        "id": "UN-003",
        "question": "What is the secret chemical formula for Pfizer's upcoming unannounced weight-loss compound?",
        "question_type": "unanswerable",
        "target_tickers": ["PFE"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    },
    {
        "id": "UN-004",
        "question": "What was the exact personal cell phone number of Tesla's CEO listed in the 10-K?",
        "question_type": "unanswerable",
        "target_tickers": ["TSLA"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    },
    {
        "id": "UN-005",
        "question": "What are the exact GPS coordinates of ExxonMobil's classified future exploration wells in 2035?",
        "question_type": "unanswerable",
        "target_tickers": ["XOM"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    },
    {
        "id": "UN-006",
        "question": "How many shoplifting incidents occurred specifically at the Walmart Supercenter in Dallas on Christmas Eve?",
        "question_type": "unanswerable",
        "target_tickers": ["WMT"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    },
    {
        "id": "UN-007",
        "question": "What is the exact percentage of Meta advertising spend coming exclusively from Martian corporations?",
        "question_type": "unanswerable",
        "target_tickers": ["META"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    },
    {
        "id": "UN-008",
        "question": "What was the internal temperature of NVIDIA's DGX SuperPOD cluster at 3:15 AM yesterday?",
        "question_type": "unanswerable",
        "target_tickers": ["NVDA"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    },
    {
        "id": "UN-009",
        "question": "What is JPMorgan Chase's unreleased internal prediction for Bitcoin's price on December 31, 2030?",
        "question_type": "unanswerable",
        "target_tickers": ["JPM"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    },
    {
        "id": "UN-010",
        "question": "What was the brand of coffee beans consumed in Amazon's board meeting room during the Q3 earnings prep?",
        "question_type": "unanswerable",
        "target_tickers": ["AMZN"],
        "gold_sections": [],
        "gold_keywords": [],
        "reference_answer": "Insufficient evidence in filed SEC disclosures.",
        "is_answerable": False,
        "human_verified": True
    }
]

def load_or_create_benchmark(file_path: Path) -> List[BenchmarkQuestion]:
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [BenchmarkQuestion(**item) for item in data]
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(BENCHMARK_QUESTIONS, f, indent=2)
    return [BenchmarkQuestion(**item) for item in BENCHMARK_QUESTIONS]


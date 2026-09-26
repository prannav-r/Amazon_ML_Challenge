"""
Comprehensive Preprocessing Pipeline Validation & Inspection Suite
Runs across representative examples from all three sources (S1, S2, S3),
specifically covering all 10 categories required by Phase 2.
"""

import sys
import io
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.preprocessing import EntityPreprocessor

preprocessor = EntityPreprocessor()

print("=" * 80)
print("PHASE 2 PREPROCESSING INSPECTION & VALIDATION SUITE")
print("=" * 80)

# ==============================================================================
# TEST CASES COVERING THE 10 SPECIFIED INSPECTION CATEGORIES
# ==============================================================================

test_cases = [
    # 1. English name vs English name
    {
        "category": "1. English Name vs English Name",
        "s1": ("S1-001", "Hepner Veterinary Clinic", "308 N Main Street, Oakwood, IL", "US"),
        "s2": ("S2-001", "Hepner Veterinary  Clinic", "308C N Main Street, Oakwood Township, Illinois", "US"),
    },
    # 2. Legal suffix variation
    {
        "category": "2. Legal Suffix Variation (US & India)",
        "s1": ("S1-002", "Eastern Integrated Ashford", "100 Broadway, New York, NY", "US"),
        "s2": ("S2-002", "Eastern Integrated Ashford LLC", "100 Broadway St, New York, NY", "US"),
    },
    {
        "category": "2b. Legal Suffix Variation (India Pvt Ltd / Limited)",
        "s1": ("S1-003", "Satya Seva Samiti", "Plot 12, Vikas Marg, Delhi", "India"),
        "s2": ("S3-003", "Satya Seva Samiti Corporation", "12 Vikas Marg, Delhi", "India"),
    },
    # 3. Punctuation variation
    {
        "category": "3. Punctuation Variation",
        "s1": ("S1-004", "Gonzalez, Altman & Thomas LLC", "500 Elm Street, Dallas, TX", "US"),
        "s2": ("S2-004", "#GONZALEZALTMAN", "500 Elm St, Dallas, TX", "US"),
    },
    {
        "category": "3b. Mojibake & En-dash / Smart quotes",
        "s1": ("S1-005", "Global Lotus Construction Pvt Ltd", "A-212 Malhotra Complex, Gali No. â€“ 01, Vikas Marg, Delhi", "India"),
        "s2": ("S2-005", "Global Lotus Construction Pvt. Ltd.", "A-212 Malhotra Complex, Gali No. - 01, Vikas Marg, Delhi", "India"),
    },
    # 4. Web / Domain-style name
    {
        "category": "4. Web / Domain-Style Name",
        "s1": ("S1-006", "Berry Financial LLC", "13 Mcdavid Street, Pell City, AL", "US"),
        "s2": ("S2-006", "berryfinancial.com", "13 Mcdavid St, Pell City, Alabama", "US"),
    },
    {
        "category": "4b. URL with www and sub-brand",
        "s1": ("S1-007", "Kishori Enterprises Private Limited", "Shop 4, Station Road, Jaipur", "India"),
        "s3": ("S3-007", "kishorienterprises.com", "Shop 4, Station Rd, Jaipur", "India"),
    },
    # 5. Indic-script name vs English name
    {
        "category": "5. Indic-Script Name vs English Name (Devanagari)",
        "s1": ("S1-008", "Real Modern Food Limited", "Shop 4, Plot 19, Sector 42, Nerul, Navi Mumbai", "India"),
        "s2": ("S2-008", "रियल मॉडर्न फूड लिमिटेड", "SHP NO. G-4, PLOT NO. 19, SECTOR 42, NERUL, NAVI MUMBAI, Maharashtra", "India"),
    },
    {
        "category": "5b. Indic-Script Name vs English Name (Tamil)",
        "s1": ("S1-009", "Global Business Pvt Ltd", "No. 21 - S, Ambattur, Chennai, Tamil Nadu", "India"),
        "s2": ("S2-009", "குளோபல் பிசினஸ் பிரைவேट லிமிடெட்", "#C-21 - S, AMBATTUR, CHENNAI, Tamil Nadu", "India"),
    },
    # 6. Address abbreviation variation
    {
        "category": "6. Address Abbreviation Variation",
        "s1": ("S1-010", "Cascade Coalition", "9802 Summerton Drive, Suite 400, Bowie, MD", "US"),
        "s3": ("S3-010", "Cascade Coalition LP", "9802 Summerton Dr, Ste 400, Bowie, Maryland", "US"),
    },
    # 7. Reordered addresses
    {
        "category": "7. Reordered Addresses",
        "s1": ("S1-011", "Hotel Ventures Limited", "C-121, Meena Bakery Chauraha Near Dariyapur, Lucknow, Uttar Pradesh", "India"),
        "s2": ("S2-011", "होटल वेंचर्स लिमिटेड", "LUCKNOW, C-121, MEENA BAKERY CHAURAHA NEAR DARIYAPUR, Uttar Pradesh", "India"),
    },
    # 8. Missing addresses
    {
        "category": "8. Missing Addresses (Preserving explicit flag, no dummy string)",
        "s1": ("S1-012", "Delta Logistics Corp", "1450 Airport Rd, Atlanta, GA", "US"),
        "s2": ("S2-012", "Delta Logistics", None, "US"),
        "s3": ("S3-012", "Delta Logistics Inc", "   ", "US"),
    },
    # 9. Postal / PIN examples
    {
        "category": "9. Postal / PIN Code Examples (US 5-digit, India 6-digit)",
        "s1": ("S1-013", "Austin Biotech LLC", "104 Innovation Way, Austin, TX 78701", "US"),
        "s2": ("S2-013", "Austin Biotech", "104 Innovation Way, Suite B, Austin, Texas 78701", "US"),
    },
    {
        "category": "9b. Indian PIN Code",
        "s1": ("S1-014", "Bangalore Tech Solutions Pvt Ltd", "Plot 24, Electronics City, Bengaluru, Karnataka 560100", "India"),
        "s3": ("S3-014", "Bangalore Tech Solutions", "24 Electronics City, Bangalore 560100", "India"),
    },
    # 10. French examples from test data
    {
        "category": "10. French Examples (Open-Set Country from Test Data)",
        "s1": ("S1-015", "Grain & Fils SAS", "329 Avenue de Dunkerque, Lille, Hauts-de-France", "France"),
        "s2": ("S2-015", "Grain & Fils", "Lille, 329 Av de Dunkerque, Hauts-de-France 59000", "France"),
    },
    {
        "category": "10b. French SARL and Diacritics",
        "s1": ("S1-016", "Maison de SantÃ© Generation SARL", "30 Rue Louis ThÃ©nard, Saint-Nazaire, Pays de la Loire", "France"),
        "s2": ("S3-016", "Maison de Sante Generation", "30 r Louis Thenard, Saint-Nazaire", "France"),
    },
]

for idx, tc in enumerate(test_cases, 1):
    cat = tc["category"]
    s1_rec = preprocessor.preprocess_record(*tc["s1"])
    s2_key = "s2" if "s2" in tc else "s3"
    s2_rec = preprocessor.preprocess_record(*tc[s2_key])

    print(f"\n[{cat}]")
    print(f"  --- Record 1 ({s1_rec.entity_id}) ---")
    print(f"    Raw Name          : '{s1_rec.name.raw}'")
    print(f"    Legal Stripped    : '{s1_rec.name.legal_stripped}'")
    print(f"    Alphanumeric Clean: '{s1_rec.name.alphanumeric_clean}'")
    print(f"    Alphanumeric NoLeg: '{s1_rec.name.alphanumeric_no_legal}'")
    print(f"    Tokens (No Legal) : {s1_rec.name.tokens_no_legal}")
    print(f"    Script / Non-ASCII: script={s1_rec.name.detected_script}, non_ascii={s1_rec.name.is_non_ascii}")
    print(f"    Raw Addr          : '{s1_rec.address.raw}'")
    print(f"    Standardized Addr : '{s1_rec.address.standardized}'")
    print(f"    Addr Numeric Nums : {s1_rec.address.numeric_tokens}")
    print(f"    Addr Postal Codes : {s1_rec.address.postal_codes}")
    print(f"    Addr Missing Flag : {s1_rec.address.is_missing}")

    print(f"  --- Record 2 ({s2_rec.entity_id}) ---")
    print(f"    Raw Name          : '{s2_rec.name.raw}'")
    print(f"    Legal Stripped    : '{s2_rec.name.legal_stripped}'")
    print(f"    Alphanumeric Clean: '{s2_rec.name.alphanumeric_clean}'")
    print(f"    Alphanumeric NoLeg: '{s2_rec.name.alphanumeric_no_legal}'")
    print(f"    Tokens (No Legal) : {s2_rec.name.tokens_no_legal}")
    print(f"    Script / Non-ASCII: script={s2_rec.name.detected_script}, non_ascii={s2_rec.name.is_non_ascii}")
    print(f"    Raw Addr          : '{s2_rec.address.raw}'")
    print(f"    Standardized Addr : '{s2_rec.address.standardized}'")
    print(f"    Addr Numeric Nums : {s2_rec.address.numeric_tokens}")
    print(f"    Addr Postal Codes : {s2_rec.address.postal_codes}")
    print(f"    Addr Missing Flag : {s2_rec.address.is_missing}")

    # Measure Equivalences Generated:
    name_alpha_eq = s1_rec.name.alphanumeric_no_legal == s2_rec.name.alphanumeric_no_legal
    name_token_jaccard = (
        len(s1_rec.name.tokens_no_legal_set & s2_rec.name.tokens_no_legal_set)
        / len(s1_rec.name.tokens_no_legal_set | s2_rec.name.tokens_no_legal_set)
        if (s1_rec.name.tokens_no_legal_set | s2_rec.name.tokens_no_legal_set)
        else 0.0
    )
    addr_token_jaccard = (
        len(s1_rec.address.tokens_set & s2_rec.address.tokens_set)
        / len(s1_rec.address.tokens_set | s2_rec.address.tokens_set)
        if (s1_rec.address.tokens_set | s2_rec.address.tokens_set)
        else 0.0
    )
    num_overlap = bool(s1_rec.address.numeric_tokens_set & s2_rec.address.numeric_tokens_set)
    postal_overlap = bool(s1_rec.address.postal_codes & s2_rec.address.postal_codes)

    print(f"  --> MATCH METRICS:")
    print(f"      Name Alpha Equivalence (No Legal): {name_alpha_eq}")
    print(f"      Name Token Jaccard (No Legal)    : {name_token_jaccard:.3f}")
    print(f"      Address Token Jaccard (Stdized)  : {addr_token_jaccard:.3f}")
    print(f"      Address Numeric Token Overlap    : {num_overlap}")
    print(f"      Postal / PIN Code Match          : {postal_overlap}")

# ==============================================================================
# COLLAPSE / OVER-NORMALIZATION AUDIT
# Test completely distinct real-world names to verify they NEVER collapse
# ==============================================================================

print("\n" + "=" * 80)
print("OVER-COLLAPSE AUDIT ON DISTINCT NAMES")
print("=" * 80)

distinct_pairs = [
    ("First National Bank", "National Commercial Bank"),
    ("American Express Co", "American Airlines Inc"),
    ("Royal Palace Hotel", "Royal Crown Industries"),
    ("Global Lotus Construction", "Global Lotus Logistics"),
    ("Smith & Sons Hardware", "Smith & Sons Bakery"),
    ("Delta Air Lines", "Delta Dental Plans"),
    ("Limited Edition Goods LLC", "Limited Brands Inc"),
    ("General Electric", "General Motors"),
]

collapse_detected = False
for n1, n2 in distinct_pairs:
    norm1 = preprocessor.preprocess_name(n1)
    norm2 = preprocessor.preprocess_name(n2)
    collapsed = norm1.alphanumeric_no_legal == norm2.alphanumeric_no_legal
    print(f"Distinct Pair: '{n1}' vs '{n2}'")
    print(f"  -> '{norm1.alphanumeric_no_legal}' vs '{norm2.alphanumeric_no_legal}' | Collapsed: {collapsed}")
    if collapsed:
        collapse_detected = True

if not collapse_detected:
    print("\nOVER-COLLAPSE AUDIT PASSED: Zero distinct entities collapsed into identical normalized forms!")
else:
    print("\nOVER-COLLAPSE AUDIT FAILED: Over-normalization detected!")

print("\nValidation completed successfully.")

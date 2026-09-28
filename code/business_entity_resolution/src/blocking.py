"""
High-Recall & Adaptive Candidate Generation Engine V2 (Phase 9-15)
Amazon ML Challenge 2026: Business Entity Resolution

Enhanced Blocker V2 supporting:
1. 11 Deterministic Blocking Channels (A, A2, B, E, E2, G, D, C, E3, H, I)
2. One-Time Candidate Pool Pre-Indexing for High-Throughput Streaming Inference
3. Priority Cap = 200 enforcement with deterministic tie-breaking
4. Strictly Open-Set Country Support via generic SQL equality (s.country = c.country)
"""

import duckdb
import os
import time
from typing import Dict, List, Optional, Tuple, Set, Any

UNAMBIGUOUS_LEGAL_REGEX = (
    r"(?:[,\s\-\(\[\#]+|\b)("
    r"private\s+limited\s+company|limited\s+liability\s+company|limited\s+liability\s+partnership|"
    r"public\s+limited\s+company|private\s+limited|public\s+limited|"
    r"pvt\s+ltd|pvt\s+limited|private\s+ltd|pub\s+ltd|pub\s+limited|"
    r"corporation|incorporated|limited|company|corp|inc|llc|llp|plc|ltd|co|pvt|"
    r"societe\s+a\s+responsabilite\s+limitee|societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|"
    r"societe\s+par\s+actions\s+simplifiee|entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|"
    r"societe\s+anonyme|societe\s+civile|sarlu|sasu|sarl|sas|eurl|sci|snc|sa|ei|et\s+fils|fils|"
    r"प्राइवेट\s+लिमिटेड|प्रा\.\s*लि\.|लिमिटेड|एलएलपी"
    r")(?:[,\s\.\)\]]*)$"
)

PREFIX_NOISE_REGEX = r"^(?:the\s+|dr\s+|smt\s+|m/s\s+|mr\s+|co\s+|>>\s+|#\s*)"
EXTENDED_PREFIX_REGEX = (
    r"^(?:the\s+|dr\s+|smt\s+|m/s\.?\s*|mr\s+|co\s+|>>\s+|#\s*|"
    r"formerly\s+|t/a\s+|d/b/a\s+|dba[:\s]+|c/o\s+)"
)

DOMAIN_STRIP_REGEX = r"(?:\.com|\.org|\.net|\.in|\.co\.in|\.co|\.us|\.biz|\.info|\.edu)(?:$|\s)"

ADDRESS_STOPWORDS_SQL = (
    "'street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', "
    "'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', "
    "'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near'"
)

BUSINESS_STOPWORDS_SQL = (
    "'enterprises', 'services', 'solutions', 'industries', 'group', 'holdings', 'trading', "
    "'company', 'limited', 'private', 'consultancy', 'associates', 'properties', 'technologies', "
    "'corp', 'corporation', 'inc', 'llc', 'pvt', 'ltd'"
)


class CandidateBlocker:
    """
    Next-generation high-recall candidate blocker powered by DuckDB SQL indexing.
    Supports 11 specialized blocking channels, Priority Cap = 200, and one-time preindexing.
    """

    def __init__(self, db_connection: Optional[duckdb.DuckDBPyConnection] = None):
        self.con = db_connection or duckdb.connect()
        self._preindexed = False

    def preindex_candidate_pool(self, cand_table_or_path: str, enable_enhanced_channels: bool = True):
        """
        Precomputes candidate pool index tables once across all 11 channels.
        Subsequent calls to generate_candidates run in ~1-2 seconds per batch.
        """
        t0 = time.time()
        cand_src = cand_table_or_path
        if os.path.exists(cand_table_or_path) or cand_table_or_path.endswith('.tsv') or cand_table_or_path.endswith('.csv'):
            self.con.execute(f"CREATE OR REPLACE TEMP VIEW _v_cand AS SELECT entity_id, business_name, business_address, country FROM read_csv('{cand_table_or_path}', delim='\\t', header=true, quote='', all_varchar=true);")
            cand_src = "_v_cand"

        prefix_regex = EXTENDED_PREFIX_REGEX if enable_enhanced_channels else PREFIX_NOISE_REGEX
        domain_sub = f"regexp_replace(replace(lower(trim(business_name)), '.', ''), '{DOMAIN_STRIP_REGEX}', '', 'g')" if enable_enhanced_channels else "replace(lower(trim(business_name)), '.', '')"

        # 1. Chan A
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cand_norm_a AS
        SELECT entity_id as candidate_id, country,
               regexp_replace(regexp_replace({domain_sub}, '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
        FROM {cand_src}
        WHERE business_name IS NOT NULL AND trim(business_name) != '';
        """)

        # 2. Chan A2
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cand_norm_a2 AS
        SELECT entity_id as candidate_id, country,
               regexp_replace(
                   regexp_replace(
                       regexp_replace({domain_sub}, '{prefix_regex}', '', 'g'),
                       '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'
                   ),
                   '[^a-z0-9]', '', 'g'
               ) as norm_key
        FROM {cand_src}
        WHERE business_name IS NOT NULL AND trim(business_name) != '';
        """)

        # 3. Chan B
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cand_name_toks AS
        SELECT country, entity_id as candidate_id, tok
        FROM (
            SELECT country, entity_id,
                   unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
            FROM {cand_src}
            WHERE business_name IS NOT NULL AND trim(business_name) != ''
        )
        WHERE length(tok) >= 4;
        """)
        self.con.execute("""
        CREATE OR REPLACE TEMP TABLE _rare_name_toks AS
        SELECT country, tok
        FROM _cand_name_toks
        GROUP BY country, tok
        HAVING count(*) BETWEEN 2 AND 200;
        """)

        # 4. Chan E
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cand_num_p3 AS
        SELECT entity_id as candidate_id, country,
               cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as addr_num,
               substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
        FROM {cand_src}
        WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
          AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3;
        """)

        # 5. Chan D & E2
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cand_addr_toks AS
        SELECT country, entity_id as candidate_id, num, tok
        FROM (
            SELECT country, entity_id,
                   cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as num,
                   unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
            FROM {cand_src}
            WHERE business_address IS NOT NULL AND trim(business_address) != ''
        )
        WHERE length(tok) >= 5 AND tok NOT IN ({ADDRESS_STOPWORDS_SQL});
        """)
        self.con.execute("""
        CREATE OR REPLACE TEMP TABLE _rare_addr_toks AS
        SELECT country, tok
        FROM _cand_addr_toks
        GROUP BY country, tok
        HAVING count(*) BETWEEN 2 AND 150;
        """)

        # 6. Chan G
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cand_del_keys AS
        SELECT country, entity_id as candidate_id,
               substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
        FROM {cand_src}
        WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 7;
        """)
        self.con.execute("""
        CREATE OR REPLACE TEMP TABLE _rare_del_keys AS
        SELECT country, del_key
        FROM _cand_del_keys
        GROUP BY country, del_key
        HAVING count(*) BETWEEN 2 AND 150;
        """)

        # 7. Chan C
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cand_ngrams AS
        SELECT entity_id as candidate_id, country,
               substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as p4,
               substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as s4
        FROM {cand_src}
        WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5;
        """)

        if enable_enhanced_channels:
            # 8. Chan E3
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cand_num_street AS
            SELECT country, entity_id as candidate_id, addr_num, street_tok
            FROM (
                SELECT country, entity_id,
                       cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as addr_num,
                       unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as street_tok
                FROM {cand_src}
                WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
            )
            WHERE length(street_tok) >= 4 AND street_tok NOT IN ({ADDRESS_STOPWORDS_SQL});
            """)
            self.con.execute("""
            CREATE OR REPLACE TEMP TABLE _rare_num_street AS
            SELECT country, addr_num, street_tok
            FROM _cand_num_street
            GROUP BY country, addr_num, street_tok
            HAVING count(*) BETWEEN 2 AND 50;
            """)

            # 9. Chan H
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cand_pin_p3 AS
            SELECT entity_id as candidate_id, country,
                   regexp_extract(business_address, '\\b[0-9]{{5,6}}\\b') as pin,
                   substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
            FROM {cand_src}
            WHERE regexp_extract(business_address, '\\b[0-9]{{5,6}}\\b') != ''
              AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3;
            """)

            # 10. Chan I
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cand_tok_pairs AS
            WITH cand_words AS (
                SELECT entity_id, country,
                       list_distinct(
                           list_filter(
                               string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' '),
                               x -> length(x) >= 4 AND x NOT IN ({BUSINESS_STOPWORDS_SQL})
                           )
                       ) as words
                FROM {cand_src}
                WHERE business_name IS NOT NULL AND trim(business_name) != ''
            ),
            cand_filtered AS (
                SELECT entity_id, country, words
                FROM cand_words
                WHERE len(words) >= 2 AND len(words) <= 6
            )
            SELECT entity_id as candidate_id, country,
                   least(w1, w2) as tok1, greatest(w1, w2) as tok2
            FROM cand_filtered,
                 UNNEST(words) AS t(w1),
                 UNNEST(words) AS u(w2)
            WHERE w1 < w2;
            """)
            self.con.execute("""
            CREATE OR REPLACE TEMP TABLE _rare_tok_pairs AS
            SELECT country, tok1, tok2
            FROM _cand_tok_pairs
            GROUP BY country, tok1, tok2
            HAVING count(*) BETWEEN 2 AND 50;
            """)

        self._preindexed = True
        print(f"Pre-indexed all 11 candidate channels in {time.time() - t0:.2f}s.")

    def generate_candidates(
        self,
        s1_table_or_path: str,
        cand_table_or_path: str,
        output_table: str = "final_candidates_v2",
        max_candidates_per_s1: int = 200,
        enable_enhanced_channels: bool = True,
        adaptive_rule: Optional[str] = None,
        adaptive_cap: int = 250,
    ) -> Dict[str, Any]:
        """
        Executes candidate generation across 11 channels and performs priority pruning.
        """
        t_start = time.time()

        s1_src = s1_table_or_path
        if os.path.exists(s1_table_or_path) or s1_table_or_path.endswith('.tsv') or s1_table_or_path.endswith('.csv'):
            self.con.execute(f"CREATE OR REPLACE TEMP VIEW _v_s1 AS SELECT entity_id, business_name, business_address, country FROM read_csv('{s1_table_or_path}', delim='\\t', header=true, quote='', all_varchar=true);")
            s1_src = "_v_s1"

        if not self._preindexed:
            self.preindex_candidate_pool(cand_table_or_path, enable_enhanced_channels=enable_enhanced_channels)

        prefix_regex = EXTENDED_PREFIX_REGEX if enable_enhanced_channels else PREFIX_NOISE_REGEX
        domain_sub = f"regexp_replace(replace(lower(trim(business_name)), '.', ''), '{DOMAIN_STRIP_REGEX}', '', 'g')" if enable_enhanced_channels else "replace(lower(trim(business_name)), '.', '')"

        # 1. CHANNEL A: Exact Core Name (Priority: 100)
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _chan_a AS
        WITH s1_clean AS (
            SELECT 
                entity_id as s1_id, country,
                regexp_replace(
                    regexp_replace({domain_sub}, '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'),
                    '[^a-z0-9]', '', 'g'
                ) as norm_key
            FROM {s1_src}
            WHERE business_name IS NOT NULL AND trim(business_name) != ''
        )
        SELECT s.s1_id, c.candidate_id, 100 as priority_score
        FROM s1_clean s
        JOIN _cand_norm_a c ON s.country = c.country AND s.norm_key = c.norm_key
        WHERE length(s.norm_key) >= 3;
        """)

        # 2. CHANNEL A2: Prefix-Stripped Core Name (Priority: 95)
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _chan_a2 AS
        WITH s1_clean AS (
            SELECT 
                entity_id as s1_id, country,
                regexp_replace(
                    regexp_replace(
                        regexp_replace({domain_sub}, '{prefix_regex}', '', 'g'),
                        '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'
                    ),
                    '[^a-z0-9]', '', 'g'
                ) as norm_key
            FROM {s1_src}
            WHERE business_name IS NOT NULL AND trim(business_name) != ''
        )
        SELECT s.s1_id, c.candidate_id, 95 as priority_score
        FROM s1_clean s
        JOIN _cand_norm_a2 c ON s.country = c.country AND s.norm_key = c.norm_key
        WHERE length(s.norm_key) >= 3;
        """)

        # 3. CHANNEL B: Rare Name Tokens (DF <= 200, Priority: 80)
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _chan_b AS
        WITH s1_tokens AS (
            SELECT country, entity_id as s1_id,
                   unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
            FROM {s1_src}
            WHERE business_name IS NOT NULL AND trim(business_name) != ''
        )
        SELECT DISTINCT s.s1_id, c.candidate_id, 80 as priority_score
        FROM s1_tokens s
        JOIN _rare_name_toks r ON s.country = r.country AND s.tok = r.tok
        JOIN _cand_name_toks c ON s.country = c.country AND s.tok = c.tok;
        """)

        # 4. CHANNEL E: Address Number + Name Prefix-3 (Priority: 75)
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _chan_e AS
        WITH s1_num AS (
            SELECT 
                entity_id as s1_id, country,
                cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as addr_num,
                substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
            FROM {s1_src}
            WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
              AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
        )
        SELECT s.s1_id, c.candidate_id, 75 as priority_score
        FROM s1_num s
        JOIN _cand_num_p3 c ON s.country = c.country AND s.addr_num = c.addr_num AND s.p3 = c.p3
        WHERE length(s.addr_num) > 0;
        """)

        # 5. CHANNEL D: Distinctive Address Tokens (DF <= 150, Priority: 50)
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _chan_d AS
        WITH s1_tokens AS (
            SELECT country, entity_id as s1_id,
                   unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
            FROM {s1_src}
            WHERE business_address IS NOT NULL AND trim(business_address) != ''
        )
        SELECT DISTINCT s.s1_id, c.candidate_id, 50 as priority_score
        FROM s1_tokens s
        JOIN _rare_addr_toks r ON s.country = r.country AND s.tok = r.tok
        JOIN _cand_addr_toks c ON s.country = c.country AND s.tok = c.tok;
        """)

        # 6. CHANNEL E2: Address Number + Distinctive Locality Anchor (Priority: 85)
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _chan_e2 AS
        WITH s1_addr_anchor AS (
            SELECT 
                entity_id as s1_id, country,
                cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as num,
                unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
            FROM {s1_src} 
            WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
        )
        SELECT DISTINCT s.s1_id, c.candidate_id, 85 as priority_score
        FROM s1_addr_anchor s
        JOIN _rare_addr_toks r ON s.country = r.country AND s.tok = r.tok
        JOIN _cand_addr_toks c ON s.country = c.country AND s.tok = c.tok AND s.num = c.num
        WHERE length(s.num) > 0;
        """)

        # 7. CHANNEL G: Frequency-Capped 1-Edit Initial-Char Typo Key (DF <= 150, Priority: 60)
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _chan_g AS
        WITH s1_del AS (
            SELECT entity_id as s1_id, country,
                   substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
            FROM {s1_src}
            WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 7
        )
        SELECT s.s1_id, c.candidate_id, 60 as priority_score
        FROM s1_del s
        JOIN _rare_del_keys r ON s.country = r.country AND s.del_key = r.del_key
        JOIN _cand_del_keys c ON s.country = c.country AND s.del_key = c.del_key;
        """)

        # 8. CHANNEL C: Character 4-gram Prefix+Suffix Inverted Index (Priority: 40)
        self.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _chan_c AS
        WITH s1_ngrams AS (
            SELECT 
                entity_id as s1_id, country,
                substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as p4,
                substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as s4
            FROM {s1_src}
            WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5
        )
        SELECT s.s1_id, c.candidate_id, 40 as priority_score
        FROM s1_ngrams s
        JOIN _cand_ngrams c ON s.country = c.country AND s.p4 = c.p4 AND s.s4 = c.s4;
        """)

        if enable_enhanced_channels:
            # 9. CHANNEL E3: Address Number + Street Token (Priority: 70)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _chan_e3 AS
            WITH s1_num_street AS (
                SELECT 
                    entity_id as s1_id, country,
                    cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as num,
                    unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as street_tok
                FROM {s1_src}
                WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
            )
            SELECT DISTINCT s.s1_id, c.candidate_id, 70 as priority_score
            FROM s1_num_street s
            JOIN _rare_num_street r ON s.country = r.country AND s.num = r.addr_num AND s.street_tok = r.street_tok
            JOIN _cand_num_street c ON s.country = c.country AND s.num = c.addr_num AND s.street_tok = c.street_tok;
            """)

            # 10. CHANNEL H: Postal Code + 3-Char Name Prefix (Priority: 65)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _chan_h AS
            WITH s1_post AS (
                SELECT 
                    entity_id as s1_id, country,
                    regexp_extract(business_address, '\\b[0-9]{{5,6}}\\b') as pin,
                    substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
                FROM {s1_src}
                WHERE business_address IS NOT NULL AND regexp_extract(business_address, '\\b[0-9]{{5,6}}\\b') != ''
                  AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
            )
            SELECT DISTINCT s.s1_id, c.candidate_id, 65 as priority_score
            FROM s1_post s
            JOIN _cand_pin_p3 c ON s.country = c.country AND s.pin = c.pin AND s.p3 = c.p3;
            """)

            # 11. CHANNEL I: Distinctive Name Token Pairs (Priority: 55)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _chan_i AS
            WITH s1_name_pairs AS (
                WITH toks AS (
                    SELECT country, entity_id as s1_id,
                           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
                    FROM {s1_src}
                    WHERE business_name IS NOT NULL AND trim(business_name) != ''
                )
                SELECT country, s1_id, tok
                FROM toks
                WHERE length(tok) >= 4 AND tok NOT IN ({BUSINESS_STOPWORDS_SQL})
            ),
            s1_tok_pairs AS (
                SELECT 
                    t1.country, t1.s1_id,
                    least(t1.tok, t2.tok) as tok_a,
                    greatest(t1.tok, t2.tok) as tok_b
                FROM s1_name_pairs t1
                JOIN s1_name_pairs t2 ON t1.s1_id = t2.s1_id AND t1.tok < t2.tok
            )
            SELECT DISTINCT s.s1_id, c.candidate_id, 55 as priority_score
            FROM s1_tok_pairs s
            JOIN _rare_tok_pairs r ON s.country = r.country AND s.tok_a = r.tok1 AND s.tok_b = r.tok2
            JOIN _cand_tok_pairs c ON s.country = c.country AND s.tok_a = c.tok1 AND s.tok_b = c.tok2;
            """)

        # UNION OF CANDIDATES & MULTI-SIGNAL SCORING
        union_queries = [
            "SELECT s1_id, candidate_id, priority_score, 'a' as chan FROM _chan_a",
            "SELECT s1_id, candidate_id, priority_score, 'a2' as chan FROM _chan_a2",
            "SELECT s1_id, candidate_id, priority_score, 'b' as chan FROM _chan_b",
            "SELECT s1_id, candidate_id, priority_score, 'e' as chan FROM _chan_e",
            "SELECT s1_id, candidate_id, priority_score, 'e2' as chan FROM _chan_e2",
            "SELECT s1_id, candidate_id, priority_score, 'g' as chan FROM _chan_g",
            "SELECT s1_id, candidate_id, priority_score, 'd' as chan FROM _chan_d",
            "SELECT s1_id, candidate_id, priority_score, 'c' as chan FROM _chan_c",
        ]
        if enable_enhanced_channels:
            union_queries.extend([
                "SELECT s1_id, candidate_id, priority_score, 'e3' as chan FROM _chan_e3",
                "SELECT s1_id, candidate_id, priority_score, 'h' as chan FROM _chan_h",
                "SELECT s1_id, candidate_id, priority_score, 'i' as chan FROM _chan_i",
            ])

        union_sql = " UNION ALL ".join(union_queries)

        cap_condition_sql = f"{max_candidates_per_s1}"

        self.con.execute(f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH all_channel_pairs AS (
            {union_sql}
        ),
        aggregated AS (
            SELECT 
                s1_id, 
                candidate_id, 
                sum(priority_score) as total_priority,
                count(DISTINCT chan) as channels_fired,
                max(CASE WHEN chan = 'a' THEN 1 ELSE 0 END) as fired_chan_a,
                max(CASE WHEN chan = 'a2' THEN 1 ELSE 0 END) as fired_chan_a2,
                max(CASE WHEN chan = 'b' THEN 1 ELSE 0 END) as fired_chan_b,
                max(CASE WHEN chan = 'c' THEN 1 ELSE 0 END) as fired_chan_c,
                max(CASE WHEN chan = 'd' THEN 1 ELSE 0 END) as fired_chan_d,
                max(CASE WHEN chan = 'e' THEN 1 ELSE 0 END) as fired_chan_e,
                max(CASE WHEN chan = 'e2' THEN 1 ELSE 0 END) as fired_chan_e2,
                max(CASE WHEN chan = 'g' THEN 1 ELSE 0 END) as fired_chan_g
            FROM all_channel_pairs
            GROUP BY s1_id, candidate_id
        ),
        ranked AS (
            SELECT 
                agg.s1_id as source1_entity_id,
                agg.candidate_id as candidate_entity_id,
                agg.total_priority,
                agg.channels_fired,
                agg.fired_chan_a,
                agg.fired_chan_a2,
                agg.fired_chan_b,
                agg.fired_chan_c,
                agg.fired_chan_d,
                agg.fired_chan_e,
                agg.fired_chan_e2,
                agg.fired_chan_g,
                row_number() OVER (
                    PARTITION BY agg.s1_id 
                    ORDER BY agg.total_priority DESC, agg.channels_fired DESC, agg.candidate_id
                ) as rank_order
            FROM aggregated agg
        )
        SELECT 
            r.source1_entity_id, r.candidate_entity_id, r.total_priority, r.channels_fired, r.rank_order,
            r.fired_chan_a, r.fired_chan_a2, r.fired_chan_b, r.fired_chan_c, r.fired_chan_d, r.fired_chan_e, r.fired_chan_e2, r.fired_chan_g
        FROM ranked r
        WHERE r.rank_order <= ({cap_condition_sql});
        """)

        # Clean up batch channel temp tables
        cleanup_tables = [
            "_chan_a", "_chan_a2", "_chan_b", "_chan_e", "_chan_e2", "_chan_g", "_chan_d", "_chan_c"
        ]
        if enable_enhanced_channels:
            cleanup_tables.extend(["_chan_e3", "_chan_h", "_chan_i"])

        for tbl in cleanup_tables:
            self.con.execute(f"DROP TABLE IF EXISTS {tbl};")

        elapsed = time.time() - t_start
        total_generated = self.con.execute(f"SELECT count(*) FROM {output_table}").fetchone()[0]

        return {
            "output_table": output_table,
            "total_candidates": total_generated,
            "runtime_seconds": elapsed,
            "max_candidates_per_s1": max_candidates_per_s1,
        }

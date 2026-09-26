import duckdb
import os
import time

base_train = "student_resource/dataset/train"
base_test = "student_resource/dataset/test"

print("--- Counting rows and basic stats ---")
con = duckdb.connect()

train_files = {
    "train_s1": f"{base_train}/train_source1.tsv",
    "train_s2": f"{base_train}/train_source2.tsv",
    "train_s3": f"{base_train}/train_source3.tsv",
    "train_gt": f"{base_train}/train_ground_truth.tsv",
}

test_files = {
    "test_s1": f"{base_test}/test_source1.tsv",
    "test_s2": f"{base_test}/test_source2.tsv",
    "test_s3": f"{base_test}/test_source3.tsv",
}

for name, path in train_files.items():
    t0 = time.time()
    count = con.execute(f"SELECT count(*) FROM read_csv('{path}', delim='\\t', header=true, quote='')").fetchone()[0]
    print(f"{name}: {count:,} rows in {time.time()-t0:.2f}s")

for name, path in test_files.items():
    t0 = time.time()
    count = con.execute(f"SELECT count(*) FROM read_csv('{path}', delim='\\t', header=true, quote='')").fetchone()[0]
    print(f"{name}: {count:,} rows in {time.time()-t0:.2f}s")

# Let's inspect country distributions in train and test
print("\n--- Train S1 Country Distribution ---")
print(con.execute(f"SELECT country, count(*) as cnt FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='') GROUP BY country").fetchall())

print("\n--- Train S2 Country Distribution ---")
print(con.execute(f"SELECT country, count(*) as cnt FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='') GROUP BY country").fetchall())

print("\n--- Train S3 Country Distribution ---")
print(con.execute(f"SELECT country, count(*) as cnt FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='') GROUP BY country").fetchall())

print("\n--- Test S1 Country Distribution ---")
print(con.execute(f"SELECT country, count(*) as cnt FROM read_csv('{base_test}/test_source1.tsv', delim='\\t', header=true, quote='') GROUP BY country").fetchall())

print("\n--- Test S2 Country Distribution ---")
print(con.execute(f"SELECT country, count(*) as cnt FROM read_csv('{base_test}/test_source2.tsv', delim='\\t', header=true, quote='') GROUP BY country").fetchall())

print("\n--- Test S3 Country Distribution ---")
print(con.execute(f"SELECT country, count(*) as cnt FROM read_csv('{base_test}/test_source3.tsv', delim='\\t', header=true, quote='') GROUP BY country").fetchall())

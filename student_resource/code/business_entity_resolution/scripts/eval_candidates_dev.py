"""Candidate scorecard on dev_eval: oracle macro F0.5 (truth-in-candidates,
P=1, empty prediction for entities with no retrieved truth), link recall,
complete coverage, row counts, with slices. Candidate sets are given as
parquet files with s1_entity_id/target_entity_id (+ optional rank columns
used for cutoffs)."""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[4]
D = (ROOT / "student_resource/dataset/train").as_posix()


def connect():
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"""
    create table split as select * from read_csv('{(ROOT/'experiments/splits/phase3_split_v1.tsv').as_posix()}', delim='\t', header=true, all_varchar=true);
    create table dev as select source1_entity_id s1id from split where phase3_split='dev_eval';
    create table gt as select source1_entity_id s1id, unnest(string_split(matched_entity_ids, ',')) tid
      from read_csv('{D}/train_ground_truth.tsv', delim='\t', header=true, quote='', all_varchar=true) where matched_entity_ids<>'';
    create table dgt as select gt.* from gt join dev using(s1id);
    create table s1c as select entity_id s1id, country from read_csv('{D}/train_source1.tsv', delim='\t', header=true, quote='', all_varchar=true);
    create table tinfo as select entity_id tid, business_address='' as addr_missing from read_csv('{D}/train_source2.tsv', delim='\t', header=true, quote='', all_varchar=true)
      union all select entity_id, business_address='' from read_csv('{D}/train_source3.tsv', delim='\t', header=true, quote='', all_varchar=true);
    create table mult as select dev.s1id, count(dgt.tid) m from dev left join dgt using(s1id) group by 1;
    """)
    return con


def score(con, cand_sql: str, label: str):
    """cand_sql: query returning distinct (s1id, tid) restricted to dev S1."""
    con.execute(f"create or replace temp table c as select distinct s1id, tid from ({cand_sql})")
    con.execute("create or replace temp table hit as select dgt.s1id, dgt.tid from dgt join c using(s1id, tid)")
    con.execute("""create or replace temp table ent as
      select mult.s1id, mult.m, coalesce(h.k,0) k, s1c.country from mult left join (select s1id, count(*) k from hit group by 1) h using(s1id)
      join s1c using(s1id)""")
    r = con.execute("""select avg(case when m=0 then 1.0 when k=0 then 0.0 else 1.25*(k/m)/(0.25+k/m) end) oracle,
       sum(k)/sum(m) recall, avg(case when m>0 then (k=m)::int end) complete, (select count(*) from c) n_rows
       from ent""").fetchone()
    sl = con.execute("""select country, avg(case when m=0 then 1.0 when k=0 then 0.0 else 1.25*(k/m)/(0.25+k/m) end), sum(k)/sum(m) from ent group by 1 order by 1""").fetchall()
    ms = con.execute("""select least(m,4) mb, avg(case when m=0 then 1.0 when k=0 then 0.0 else 1.25*(k/m)/(0.25+k/m) end), count(*) from ent group by 1 order by 1""").fetchall()
    src = con.execute("""select substr(dgt.tid,1,2), avg((hit.tid is not null)::int) from dgt left join hit using(s1id,tid) group by 1 order by 1""").fetchall()
    am = con.execute("""select tinfo.addr_missing, avg((hit.tid is not null)::int), count(*) from dgt join tinfo using(tid) left join hit using(s1id,tid) group by 1 order by 1""").fetchall()
    print(f"{label:40s} oracle={r[0]:.6f} recall={r[1]:.4f} complete={r[2]:.4f} rows={r[3]:,}")
    print(f"   country={[(a, round(b,4), round(c,4)) for a,b,c in sl]} mult={[(a, round(b,4)) for a,b,_ in ms]} src_recall={[(a, round(b,4)) for a,b in src]} addr_missing_recall={[(a, round(b,4)) for a,b,_ in am]}")
    return r


if __name__ == "__main__":
    con = connect()
    for arg in sys.argv[1:]:
        score(con, arg, arg[:40])

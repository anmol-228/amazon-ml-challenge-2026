"""Compare two candidate sets on dev_eval (full population): oracle macro F0.5,
link recall, complete-entity coverage, country slices, residual classes
(script-named target, target address missing), marginal rescue and cost.

Usage: python compare_candidates_dev.py TAG_BASE TAG_NEW
(tags are experiments/r1/<tag>/cand_*.parquet)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[4]
D = (ROOT / "student_resource/dataset/train").as_posix()
SCRIPT_CLASS = "[" + chr(0x0900) + "-" + chr(0x0DFF) + "]"


def main() -> None:
    base, new = sys.argv[1], sys.argv[2]
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"""
    create table split as select * from read_csv('{(ROOT/'experiments/splits/phase3_split_v1.tsv').as_posix()}', delim='\t', header=true, all_varchar=true);
    create table dev as select source1_entity_id s1id from split where phase3_split='dev_eval';
    create table gt as select source1_entity_id s1id, unnest(string_split(matched_entity_ids, ',')) tid
      from read_csv('{D}/train_ground_truth.tsv', delim='\t', header=true, quote='', all_varchar=true) where matched_entity_ids<>'';
    create table dgt as select gt.* from gt join dev using(s1id);
    create table s1c as select entity_id s1id, country from read_csv('{D}/train_source1.tsv', delim='\t', header=true, quote='', all_varchar=true);
    create table tinfo as select entity_id tid, business_address is null as addr_missing,
        regexp_matches(business_name, '{SCRIPT_CLASS}') as script_name
      from (select * from read_csv('{D}/train_source2.tsv', delim='\t', header=true, quote='', all_varchar=true)
            union all select * from read_csv('{D}/train_source3.tsv', delim='\t', header=true, quote='', all_varchar=true));
    create table mult as select dev.s1id, count(dgt.tid) m from dev left join dgt using(s1id) group by 1;
    """)
    out = {}
    for tag, name in ((base, "base"), (new, "new")):
        glob = (ROOT / "experiments" / "r1" / tag / "cand_*.parquet").as_posix()
        con.execute(f"""create table c_{name} as select distinct s1_entity_id s1id, target_entity_id tid
                        from read_parquet('{glob}') where s1_entity_id in (select s1id from dev)""")
        con.execute(f"create table h_{name} as select dgt.s1id, dgt.tid from dgt join c_{name} using(s1id, tid)")
        r = con.execute(f"""
          with ent as (select mult.s1id, mult.m, coalesce(h.k,0) k, s1c.country from mult
                       left join (select s1id, count(*) k from h_{name} group by 1) h using(s1id) join s1c using(s1id))
          select avg(case when m=0 then 1.0 when k=0 then 0.0 else 1.25*(k/m)/(0.25+k/m) end),
                 sum(k)/sum(m), avg(case when m>0 then (k=m)::int end),
                 avg(case when m=0 then 1.0 when k=0 then 0.0 else 1.25*(k/m)/(0.25+k/m) end) filter (where country='US'),
                 avg(case when m=0 then 1.0 when k=0 then 0.0 else 1.25*(k/m)/(0.25+k/m) end) filter (where country='India')
          from ent""").fetchone()
        miss = con.execute(f"""select sum((tinfo.script_name and not tinfo.addr_missing)::int), sum(tinfo.addr_missing::int), count(*)
                               from dgt join tinfo using(tid) anti join h_{name} using(s1id, tid)""").fetchone()
        rows = con.execute(f"select count(*) from c_{name}").fetchone()[0]
        out[name] = {"tag": tag, "oracle": r[0], "link_recall": r[1], "complete": r[2], "oracle_US": r[3],
                     "oracle_India": r[4], "rows": rows, "missed_links": miss[2],
                     "missed_script_name_addr_present": miss[0], "missed_target_addr_missing": miss[1]}
    rescued = con.execute("select count(*) from h_new anti join h_base using(s1id, tid)").fetchone()[0]
    lost = con.execute("select count(*) from h_base anti join h_new using(s1id, tid)").fetchone()[0]
    b, n = out["base"], out["new"]
    out["delta"] = {k: n[k] - b[k] for k in ("oracle", "link_recall", "complete", "oracle_US", "oracle_India", "rows",
                                             "missed_links", "missed_script_name_addr_present", "missed_target_addr_missing")}
    out["delta"]["rescued_links"] = rescued
    out["delta"]["lost_links"] = lost
    out["delta"]["rows_per_net_rescued"] = (n["rows"] - b["rows"]) / max(rescued - lost, 1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

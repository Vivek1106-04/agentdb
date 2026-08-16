-- TPC-H on ClickHouse — the local half of the cross-engine suite (SPEC §11.2).
--
-- Deliberately NOT in docker-entrypoint-initdb.d: everything in that directory
-- runs on first container start, and a stranger running `make up` should get a
-- warehouse in seconds, not one that blocks on 30M rows. `make load-tpch`
-- applies this file and then loads the data.
--
-- Column names, order, and decimal precision mirror Databricks `samples.tpch`
-- exactly, verified against `samples.information_schema.columns` on 2026-08-16:
-- keys are 64-bit, the four counted columns (p_size, ps_availqty,
-- o_shippriority, l_linenumber) are 32-bit, and every money column is
-- DECIMAL(18,2) — not the DECIMAL(15,2) of the TPC-H specification. Matching the
-- precision is not pedantry: Spark and ClickHouse both derive a product's scale
-- from its operands', so a width mismatch would make
-- `SUM(l_extendedprice * (1 - l_discount))` round differently on the two engines
-- and turn a grounding result into an arithmetic artifact.
--
-- One asymmetry is not repairable and is therefore recorded instead. Delta
-- declares every column nullable; these declare none. TPC-H data contains no
-- NULLs, so results agree — but the A0 arm's schema dump reads differently per
-- engine, and `suites/tpch_nl/README.md` says so rather than hiding it.

CREATE DATABASE IF NOT EXISTS tpch;

-- Sort keys are the primary keys, which is what ClickHouse's own TPC-H
-- documentation uses. That choice is what makes this suite worth running: the
-- questions people actually ask filter on l_shipdate and o_orderdate, neither of
-- which is a sort-key prefix, so the primary index cannot prune them. The
-- failure mode of SPEC §7 is therefore present in the data by construction, not
-- simulated.

CREATE TABLE IF NOT EXISTS tpch.region (
    r_regionkey Int64,
    r_name      String,
    r_comment   String
) ENGINE = MergeTree ORDER BY r_regionkey;

CREATE TABLE IF NOT EXISTS tpch.nation (
    n_nationkey Int64,
    n_name      String,
    n_regionkey Int64,
    n_comment   String
) ENGINE = MergeTree ORDER BY n_nationkey;

CREATE TABLE IF NOT EXISTS tpch.supplier (
    s_suppkey   Int64,
    s_name      String,
    s_address   String,
    s_nationkey Int64,
    s_phone     String,
    s_acctbal   Decimal(18, 2),
    s_comment   String
) ENGINE = MergeTree ORDER BY s_suppkey;

CREATE TABLE IF NOT EXISTS tpch.customer (
    c_custkey    Int64,
    c_name       String,
    c_address    String,
    c_nationkey  Int64,
    c_phone      String,
    c_acctbal    Decimal(18, 2),
    c_mktsegment String,
    c_comment    String
) ENGINE = MergeTree ORDER BY c_custkey;

CREATE TABLE IF NOT EXISTS tpch.part (
    p_partkey     Int64,
    p_name        String,
    p_mfgr        String,
    p_brand       String,
    p_type        String,
    p_size        Int32,
    p_container   String,
    p_retailprice Decimal(18, 2),
    p_comment     String
) ENGINE = MergeTree ORDER BY p_partkey;

CREATE TABLE IF NOT EXISTS tpch.partsupp (
    ps_partkey    Int64,
    ps_suppkey    Int64,
    ps_availqty   Int32,
    ps_supplycost Decimal(18, 2),
    ps_comment    String
) ENGINE = MergeTree ORDER BY (ps_partkey, ps_suppkey);

CREATE TABLE IF NOT EXISTS tpch.orders (
    o_orderkey      Int64,
    o_custkey       Int64,
    o_orderstatus   String,
    o_totalprice    Decimal(18, 2),
    o_orderdate     Date,
    o_orderpriority String,
    o_clerk         String,
    o_shippriority  Int32,
    o_comment       String
) ENGINE = MergeTree ORDER BY o_orderkey;

CREATE TABLE IF NOT EXISTS tpch.lineitem (
    l_orderkey      Int64,
    l_partkey       Int64,
    l_suppkey       Int64,
    l_linenumber    Int32,
    l_quantity      Decimal(18, 2),
    l_extendedprice Decimal(18, 2),
    l_discount      Decimal(18, 2),
    l_tax           Decimal(18, 2),
    l_returnflag    String,
    l_linestatus    String,
    l_shipdate      Date,
    l_commitdate    Date,
    l_receiptdate   Date,
    l_shipinstruct  String,
    l_shipmode      String,
    l_comment       String
) ENGINE = MergeTree ORDER BY (l_orderkey, l_linenumber);

-- The harness authenticates as agentdb_ro (SPEC §13.3). Without this grant the
-- tables exist and every measured query fails with an access error.
GRANT SELECT ON tpch.* TO agentdb_ro;

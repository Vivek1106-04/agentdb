-- A reference workload for the TPC-H schema, in the shapes the benchmark's own
-- 22 queries use.
--
-- Same provenance argument as clickbench.sql: TPC-H predates this project by
-- three decades and its query shapes were not chosen by us. Arm A6's demand
-- signal has to come from somewhere that is neither this project's task suite
-- (which would be measuring our own authoring) nor the live query log (which on
-- a benchmark instance holds this project's own gold executions).
--
-- Literals are representative rather than randomized: the advisor reads which
-- columns are filtered, grouped and joined, not which constants appear.

SELECT l_returnflag, l_linestatus, SUM(l_quantity), SUM(l_extendedprice), AVG(l_discount), COUNT(*)
FROM lineitem WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey
AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC LIMIT 10;

SELECT o_orderpriority, COUNT(*) AS order_count FROM orders
WHERE o_orderdate >= DATE '1993-07-01' AND o_orderdate < DATE '1993-10-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey
AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
AND r_name = 'ASIA' AND o_orderdate >= DATE '1994-01-01' AND o_orderdate < DATE '1995-01-01'
GROUP BY n_name ORDER BY revenue DESC;

SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24;

SELECT l_shipmode, COUNT(*) FROM orders, lineitem
WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP')
AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
AND l_receiptdate >= DATE '1994-01-01' AND l_receiptdate < DATE '1995-01-01'
GROUP BY l_shipmode ORDER BY l_shipmode;

SELECT c_count, COUNT(*) AS custdist FROM (
  SELECT c_custkey, COUNT(o_orderkey) AS c_count FROM customer LEFT OUTER JOIN orders
  ON c_custkey = o_custkey GROUP BY c_custkey
) AS c_orders GROUP BY c_count ORDER BY custdist DESC;

SELECT 100.00 * SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM lineitem, part WHERE l_partkey = p_partkey
AND l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01';

SELECT s_name, s_address FROM supplier, nation
WHERE s_nationkey = n_nationkey AND n_name = 'CANADA' ORDER BY s_name;

SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue, c_acctbal, n_name
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01'
AND l_returnflag = 'R' AND c_nationkey = n_nationkey
GROUP BY c_custkey, c_name, c_acctbal, n_name ORDER BY revenue DESC LIMIT 20;

SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value
FROM partsupp, supplier, nation
WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
GROUP BY ps_partkey ORDER BY value DESC;

SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM partsupp, part
WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#45' AND p_size IN (49, 14, 23, 45, 19)
GROUP BY p_brand, p_type, p_size ORDER BY supplier_cnt DESC;

SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part
WHERE p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container = 'MED BOX';

SELECT o_orderpriority, o_orderdate, SUM(l_quantity) FROM customer, orders, lineitem
WHERE c_custkey = o_custkey AND o_orderkey = l_orderkey
GROUP BY o_orderpriority, o_orderdate ORDER BY o_orderdate LIMIT 100;

-- A reference workload for the ClickBench `hits` table.
--
-- These are the shapes of the published ClickBench queries, which predate this
-- project by years and were written by ClickHouse, not by us. That provenance is
-- the point: arm A6 feeds the advisor a demand signal, and a demand signal
-- authored to match this benchmark's own tasks would inflate A6 into measuring
-- our own authoring rather than the advisor.
--
-- The alternative — mining the live query log — is worse here, because on a
-- benchmark instance that log contains this project's own gold executions. In a
-- real deployment the log is the right source and `mine_workload` reads it; for
-- a measured arm it would be contamination.
--
-- Source: https://github.com/ClickHouse/ClickBench (queries.sql), shape-preserving.

SELECT COUNT(*) FROM hits;

SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;

SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;

SELECT AVG(UserID) FROM hits;

SELECT COUNT(DISTINCT UserID) FROM hits;

SELECT COUNT(DISTINCT SearchPhrase) FROM hits;

SELECT MIN(EventDate), MAX(EventDate) FROM hits;

SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;

SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;

SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT UserID)
FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10;

SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits
WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;

SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> ''
GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> ''
GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;

SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> ''
GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10;

SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;

SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase
ORDER BY COUNT(*) DESC LIMIT 10;

SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*)
FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;

SELECT UserID FROM hits WHERE UserID = 435090932899640449;

SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%';

SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits
WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;

SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;

SELECT CounterID, AVG(length(URL)) AS l, COUNT(*) AS c FROM hits
WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;

SELECT SUM(ResolutionWidth) FROM hits;

SELECT DATE_TRUNC('minute', EventTime) AS m, COUNT(*) FROM hits
WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15'
GROUP BY m ORDER BY m LIMIT 10;

SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits
WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;

SELECT URL, COUNT(*) AS c FROM hits WHERE CounterID = 62
AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0
GROUP BY URL ORDER BY c DESC LIMIT 10;

SELECT Title, COUNT(*) AS c FROM hits WHERE CounterID = 62
AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0
GROUP BY Title ORDER BY c DESC LIMIT 10;

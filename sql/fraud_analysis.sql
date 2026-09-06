-- ============================================================================
-- CREDIT CARD FRAUD DETECTION - PRODUCTION SQL ANALYTICAL QUERIES
-- ============================================================================
-- Schema:
-- Table Name: creditcard_transactions
-- Columns:
--   Time     : NUMERIC (Seconds elapsed from first transaction)
--   V1 - V28 : NUMERIC (PCA transformed confidential features)
--   Amount   : NUMERIC (Transaction amount in EUR)
--   Class    : INTEGER (0 = Legitimate, 1 = Fraudulent)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- QUERY 1: EXECUTIVE KPI SUMMARY & OVERALL FRAUD RATE
-- Objective: Calculate overall transaction volume, legitimate vs fraud counts,
--            overall fraud rate percentage, and total monetary volume.
-- ----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) AS total_legit_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS total_fraud_transactions,
    ROUND(CAST(SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100.0, 4) AS fraud_rate_percentage,
    ROUND(SUM(Amount), 2) AS total_amount_eur,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS total_fraud_amount_eur,
    ROUND(CAST(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END) AS FLOAT) / SUM(Amount) * 100.0, 4) AS fraud_amount_share_percentage
FROM creditcard_transactions;


-- ----------------------------------------------------------------------------
-- QUERY 2: TRANSACTION AMOUNT COMPARISON BY CLASS
-- Objective: Compare key descriptive statistics (Count, Average, Min, Max, 
--            Total Volume) for Legitimate vs Fraudulent transactions.
-- ----------------------------------------------------------------------------
SELECT 
    CASE WHEN Class = 1 THEN 'Fraudulent (1)' ELSE 'Legitimate (0)' END AS transaction_class,
    COUNT(*) AS transaction_count,
    ROUND(AVG(Amount), 2) AS avg_amount_eur,
    ROUND(MIN(Amount), 2) AS min_amount_eur,
    ROUND(MAX(Amount), 2) AS max_amount_eur,
    ROUND(SUM(Amount), 2) AS total_amount_eur,
    ROUND(AVG(Amount * Amount) - (AVG(Amount) * AVG(Amount)), 2) AS amount_variance
FROM creditcard_transactions
GROUP BY Class
ORDER BY Class ASC;


-- ----------------------------------------------------------------------------
-- QUERY 3: AMOUNT-CATEGORY FRAUD RATE ANALYSIS
-- Objective: Segment transactions into standard risk tiers:
--            - Low: €0 to €20
--            - Medium: €20 to €100
--            - High: €100 to €500
--            - Very High: €500+
--            Assess transaction volume, fraud rate %, and monetary exposure.
-- ----------------------------------------------------------------------------
WITH categorized_transactions AS (
    SELECT 
        Amount,
        Class,
        CASE 
            WHEN Amount <= 20.00 THEN '1. Low (€0 - €20)'
            WHEN Amount > 20.00 AND Amount <= 100.00 THEN '2. Medium (€20 - €100)'
            WHEN Amount > 100.00 AND Amount <= 500.00 THEN '3. High (€100 - €500)'
            ELSE '4. Very High (€500+)'
        END AS amount_tier
    FROM creditcard_transactions
)
SELECT 
    amount_tier,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) AS legit_transactions,
    ROUND(CAST(SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100.0, 4) AS fraud_rate_percentage,
    ROUND(SUM(Amount), 2) AS total_tier_amount_eur,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS fraud_tier_amount_eur
FROM categorized_transactions
GROUP BY amount_tier
ORDER BY amount_tier ASC;


-- ----------------------------------------------------------------------------
-- QUERY 4: 24-HOUR TEMPORAL FRAUD DISTRIBUTION (HOUR-OF-DAY)
-- Objective: Derive the hour of day from elapsed seconds: (FLOOR(Time / 3600) % 24)
--            Determine high-risk diurnal fraud patterns and vulnerability windows.
-- ----------------------------------------------------------------------------
WITH hourly_transactions AS (
    SELECT 
        CAST((FLOOR(Time / 3600.0)) AS INTEGER) % 24 AS hour_of_day,
        Amount,
        Class
    FROM creditcard_transactions
)
SELECT 
    hour_of_day,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) AS legit_transactions,
    ROUND(CAST(SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100.0, 4) AS hourly_fraud_rate_pct,
    ROUND(SUM(Amount), 2) AS total_hourly_amount_eur,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS fraud_hourly_amount_eur
FROM hourly_transactions
GROUP BY hour_of_day
ORDER BY hour_of_day ASC;


-- ----------------------------------------------------------------------------
-- QUERY 5: HIGH-RISK SUSPICIOUS TRANSACTION FLAGGING (RULE-BASED SCREENING)
-- Objective: Identify high-risk transactions exhibiting extreme features 
--            (e.g., negative outlier signals in V14, V12, V17 combined with high amounts).
-- ----------------------------------------------------------------------------
SELECT 
    Time,
    CAST((FLOOR(Time / 3600.0)) AS INTEGER) % 24 AS hour_of_day,
    Amount,
    V14,
    V12,
    V17,
    V4,
    V11,
    Class AS actual_class,
    CASE 
        WHEN (V14 < -5.0 OR V12 < -5.0 OR V17 < -5.0) AND (V4 > 3.0 OR V11 > 3.0) THEN 'CRITICAL_RISK'
        WHEN (V14 < -3.0 OR V12 < -3.0 OR V17 < -3.0) THEN 'HIGH_RISK'
        ELSE 'STANDARD'
    END AS heuristic_risk_level
FROM creditcard_transactions
WHERE Class = 1 OR (V14 < -5.0 AND V4 > 3.0)
ORDER BY Amount DESC
LIMIT 50;


-- ----------------------------------------------------------------------------
-- QUERY 6: TOP 20 LARGEST FRAUDULENT TRANSACTIONS BY LOSS IMPACT
-- Objective: Rank highest single financial losses caused by confirmed fraud.
-- ----------------------------------------------------------------------------
SELECT 
    Time,
    CAST((FLOOR(Time / 3600.0)) AS INTEGER) % 24 AS hour_of_day,
    Amount AS fraud_amount_eur,
    V1, V2, V3, V4, V14, V17
FROM creditcard_transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 20;

--1) How many rows are in the dataset?
SELECT COUNT(*) AS total_rows FROM heart_data;

--2) Average values of key columns
SELECT 
    AVG(age) AS avg_age,
    AVG(trestbps) AS avg_bp,
    AVG(chol) AS avg_chol,
    AVG(thalach) AS avg_hr,
    AVG(oldpeak) AS avg_oldpeak
FROM heart_data;

--3) Count of each heart disease severity level
SELECT target, COUNT(*) AS count 
FROM heart_data 
GROUP BY target 
ORDER BY target;

--4) Male vs Female count
SELECT 
    CASE sex 
        WHEN 0 THEN 'Female' 
        WHEN 1 THEN 'Male' 
    END AS gender,
    COUNT(*) AS count
FROM heart_data
GROUP BY sex;

--5) Average cholesterol and blood pressure by target level
SELECT 
    target,
    ROUND(AVG(chol), 1) AS avg_chol,
    ROUND(AVG(trestbps), 1) AS avg_bp
FROM heart_data
GROUP BY target
ORDER BY target;

--6) Heart disease rate by sex
SELECT 
    sex,
    target,
    COUNT(*) AS count
FROM heart_data
GROUP BY sex, target
ORDER BY sex, target;

--7) Average heart rate(thalach) by severity 
SELECT 
    target,
    ROUND(AVG(thalach), 1) AS avg_heart_rate
FROM heart_data
GROUP BY target
ORDER BY target;

--8) Heart Disease Rate by Cholesterol Range
-- Heart Disease Rate by Cholesterol Level Range
SELECT 
  CASE 
    WHEN chol < 200 THEN '< 200'
    WHEN chol BETWEEN 200 AND 239 THEN '200–239'
    WHEN chol BETWEEN 240 AND 279 THEN '240–279'
    ELSE '280+' 
  END AS chol_range,
  
  COUNT(*) AS total,
  
  SUM(CASE WHEN target > 0 THEN 1 ELSE 0 END) AS with_disease,
  
  ROUND(100.0 * SUM(CASE WHEN target > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS disease_rate_percent

FROM heart_data
GROUP BY chol_range
ORDER BY disease_rate_percent DESC;

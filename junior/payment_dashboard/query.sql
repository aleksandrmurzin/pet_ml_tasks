WITH a AS (
    SELECT cast(date_trunc('month', cast(DATE AS DATE)) AS DATE) AS date,
        id,
        amount,
        status_true,
        mode,
        email_id,
        phone_id,
        COUNT(*) OVER(
            PARTITION BY cast(date_trunc('month', cast(DATE AS DATE)) AS DATE),
            mode,
            status_true != 'Confirmed'
        ) AS success_orders,
        COUNT(*) OVER(
            PARTITION BY cast(date_trunc('month', cast(DATE AS DATE)) AS DATE),
            mode
        ) AS total_orders
    FROM (
            SELECT *,
                CASE
                    WHEN status = 'Confirmed' THEN 'Confirmed'
                    ELSE 'Other'
                END AS status_true
            FROM new_payments
        ) as new_payments
    WHERE mode != 'Не определено'
)
SELECT date,
    mode,
    status_true,
    100 - (success_orders::float / total_orders::float) * 100 as percents
FROM a
GROUP BY date,
    mode,
    status_true,
    (success_orders::float / total_orders::float)
ORDER BY date,
    mode
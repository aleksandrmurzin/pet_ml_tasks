with a as(
    select user_id,
        toStartOfDay(timestamp) as day
    from default.churn_submits
    order by user_id,
        timestamp
),
b as(
    select day as day2,
        -- substring(toString(day), 1, 10) as day,
        user_id
    from a
    group by day,
        user_id
    order by day
)
select day2 - interval('d')
from b -- select day,
    --     count(user_id) over(partition by day rows between 6 preceding and current row)
    --   from b
    -- select day,
    --     count(distinct num_users) over(
    --         rows between 6 preceding and current row
    --     ) as wau
    -- from b
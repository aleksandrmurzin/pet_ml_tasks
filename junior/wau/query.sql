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


  with a as(
select user_id,
       toStartOfDay(timestamp) as day
  from default.churn_submits
 order by timestamp,
       user_id
       b as(
select first_value(user_id) as userid,
       day,
       day - interval '7 day' as week_before
  from a
 group by user_id, day
 order by day


       )

-- select day,
--       user_id,
--       date_sub(week, 1, day) as week_before,
--       count(distinct user_id) over(partition by day)

--   from a


-- select count(user_id) over(partition by day range between day - interval '7 day' and day)
 select first_value(user_id) as userid,
        day,
        day - interval '7 day' as week_before
  from a
 group by user_id, day
 order by day
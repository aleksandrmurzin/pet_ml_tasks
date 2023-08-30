with a as(
    select cast(date_trunc('month', cast(DATE AS DATE)) AS DATE) AS date,
        status,
        mode
    from new_payments
    where mode != 'Не определено'
),
b as(
    select date,
        mode,
        count(status)::float as total_confirmed
    from a
    where status = 'Confirmed'
    group by date,
        mode
    order by date,
        mode
),
c as(
    select date,
        mode,
        count(status)::float as total
    from a
    group by date,
        mode
    order by date,
        mode
),
d as(
    select date,
        c.mode,
        b.total_confirmed,
        c.total
    from b
        full join c using (date, mode)
)
select date as time,
    mode,
    (
        case
            when total_confirmed is NULL then 0
            else total_confirmed
        end
    ) / total * 100 as percents
from d
order by date,
    mode
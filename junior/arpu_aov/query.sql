select cast(date_trunc('month', date) as date) as time,
    sum(amount) / count(distinct email_id) as arppu,
    avg(amount) as aov
from new_payments
where status = 'Confirmed'
group by cast(date_trunc('month', date) as date)
order by cast(date_trunc('month', date) as date)
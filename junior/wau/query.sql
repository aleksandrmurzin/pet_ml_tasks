  with a as(
select distinct(user_id),
       substring(toString(timestamp),1,10) as day
  from default.churn_submits
 order by substring(toString(timestamp),1,10),
       user_id),
       b as(
select distinct(toDate(day)) as day,
       count(distinct user_id) over(order by toDate(day) range between 6 preceding and current row) as wau
  from a
  )

select *
  from b
 order by day
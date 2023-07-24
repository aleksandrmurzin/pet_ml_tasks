select toDate(toStartOfDay(timestamp)) as day,
       count(distinct user_id) as dau
  from default.churn_submits
 group by toDate(toStartOfDay(timestamp))
 order by day
select month,
       avg(check_amount) as avg_check,
       quantileExactExclusive(0.5) (check_amount) as median_check
  from (select *,
              toStartOfMonth(toDateTime(buy_date)) as month
          from default.view_checks) as a
 group by month
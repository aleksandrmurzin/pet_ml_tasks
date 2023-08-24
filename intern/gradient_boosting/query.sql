  with a as(
select * except (id),
       dateDiff('day', loan_start, loan_deadline) as loan_period,
       if(dateDiff('day', loan_payed, loan_deadline) > 0, 0, -dateDiff('day', loan_payed, loan_deadline)) as delay_days
  from default.loan_delay_days
 order by id
)

select * except(loan_start, loan_deadline, loan_payed)
  from a
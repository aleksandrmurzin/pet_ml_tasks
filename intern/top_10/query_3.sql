select vendor,
       count(distinct brand) as brand
  from sku_dict_another_one
 where vendor is not null
   and brand is not null
 group by vendor
 order by brand desc
 limit 10
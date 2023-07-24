select sku_type,
       count(distinct vendor) as count_vendor
  from sku_dict_another_one
 where vendor is not null
   and sku_type is not null
 group by sku_type
 order by count_vendor desc
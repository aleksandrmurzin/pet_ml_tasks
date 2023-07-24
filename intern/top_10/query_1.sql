select brand,
       count(sku_type) as count_sku
  from sku_dict_another_one
 where brand is not null
 group by brand
 order by count_sku desc
 limit 10
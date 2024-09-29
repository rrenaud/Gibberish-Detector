library(data.table)
source("connect_to_redshift.R")
con_redshift <- connect_to_redshift()
query <- "with oc as (
  select user_id,count(distinct id) as order_count from zepto_oms_db_order where status='DELIVERED' group by 1
)
(select ua.building_name from zepto_oms_db_order o
left join zepto_oms_db_user_address ua on ua.id = o.user_address_id
left join oc on oc.user_id = o.user_id 
where date(o.created_on+interval '330' minute) >= current_date - 5
and oc.order_count >= 10)

union all

(select ua.flat_details from zepto_oms_db_order o
left join zepto_oms_db_user_address ua on ua.id = o.user_address_id
left join oc on oc.user_id = o.user_id 
where date(o.created_on+interval '330' minute) >= current_date - 5
and oc.order_count >= 10)

union all

(select ua.landmark from zepto_oms_db_order o
left join zepto_oms_db_user_address ua on ua.id = o.user_address_id
left join oc on oc.user_id = o.user_id 
where date(o.created_on+interval '330' minute) >= current_date - 5
and oc.order_count >= 10 and (ua.landmark is not null or ua.landmark != ''))
"
addresses <- data.table(dbGetQuery(con_redshift,query))
addresses <- unique(addresses)
addresses$len <- sapply(addresses$building_name, nchar)
addresses <- addresses[len>10]
fwrite(addresses,"addresses.csv")
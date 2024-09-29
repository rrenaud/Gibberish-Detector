library(DBI)
connect_to_redshift <- function() {
  db <- "dev"  #provide the name of your db
  host_db <- "redshift-readonly-cluster.c4su0dcdt8kw.ap-south-1.redshift.amazonaws.com"
  db_port <- "5434"  # or any other port specified by the DBA
  db_user <- "ro_growth_user"
  db_password <- "z%132m!3HdS3"
  #establish_connection----
  return( dbConnect(RPostgres::Postgres(),
                    dbname = db,
                    host=host_db,
                    port=db_port,
                    user=db_user,
                    password=db_password))
}

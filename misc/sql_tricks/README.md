# SQL Tricks

Here are some tricks in SQL that I found useful. It will keep updating.

### Create a table with consecutive numbers

```sql
select (@row := @row + 1) as n
from tablename t1
join (select @row := 0) t2
limit 30 -- sequence length to specify
```

There are two parameters we can tweak here to accomodate the needs:

`tablename`: Name of a table existing in the database, with row number greater or equal to the sequence lenghth that we need.

`sequence length`: The number at `limit` , currently set to 30. We can change the number to anything smaller or equal to the row number of the table we specified above.


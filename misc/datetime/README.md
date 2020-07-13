# Datetime

Some of the materials are from Corey Schafer's Python Tutorial series: [video](https://www.youtube.com/watch?v=eirjjyP2qcQ) and [GitHub page](https://github.com/CoreyMSchafer/code_snippets/blob/master/Datetime/dates.py).

Examples of using the `datetime` module can be found in the jupyter notebook that I made: [ipynb](datetime.ipynb) and [html](https://htmlpreview.github.io/?https://github.com/zhouxu-ds/DS_tools/blob/master/misc/datetime/datetime.html)

## Table of Content

- [Native Obejcts](#native)
- [timedelta Obejcts](#timedelta)
- [datetime Objects](#datetime)
- [Between Str and datetime](#string)
- [Appendix: See all timezones](#tz)

<a name='native'></a>

## Native Objects - date and time

`datetime.date` and `datetime.time` are native objects that do not include timezone information.

The date or time objects can be simply constructed like below:

```python
import datetime

# date object of a specific date
d = datetime.date(2001, 9, 11)
print(d) # 2001-09-11

# time object of a specific time point
t = datetime.time(9, 30, 20)
print(t) # 09:30:20
```

Or it can also come from current timestamp, like:

```python
# date object of today
tday = datetime.date.today()
print(tday) # 2020-07-11
```

<a name='timedelta'></a>

## timedelta Objects

`datetime.timedelta` objects represents the difference between time.

```python
# Use timedelta to get net week's date
tdelta = datetime.timedelta(days=7)
print(tday + tdelta) # 2020-07-18

# How long is my birthday from today
bday = datetime.date(2020, 9, 20)
tillday = bday - tday
print(tillday.days) # 71
```

<a name='datetime'></a>

## datetime Objects

Different from the native objects, the datetime objects are capable of integrating timezone information, which is very useful in many cases, so it is the most commonly used object type in the `datetime` module.

`pytz` is a commonly used library to handle the timezone information.

Silmilar to the usage of `datetime.date` and `datetime.time`, the timezone arguments can be included or not:

```python
import pytz

dtnow = datetime.datetime.now()
print(dtnow) # 2020-07-11 17:10:51.423325

# Get the US Eastern time now by specifying the tz argument
dtnow = datetime.datetime.now(tz=pytz.timezone('US/Eastern'))
print(dtnow) # 2020-07-11 19:10:51.674733-04:00

# Can define a datetime object by giving the tzinfo
dt = datetime.datetime(2016, 7, 24, 12, 30, 45, tzinfo=pytz.UTC)
print(dt) # 2016-07-24 12:30:45+00:00
```

**Note**: We are specifying `tz` in `now()` method, but `tzinfo` in direct constructions of `datetime` objects.

<a name='string'></a>

## Convertion Between String and datetime Objects

- ` strptime` - convert from string to datetime.
- `strftime` - convert from datetime to string.

The format code can be found [here](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) in the documentation.

```python
# Convert string to datetime
dt = datetime.datetime.strptime('Jul 18 2020', '%b %d %Y')
print(dt.date()) # 2020-07-18

# convert datetime to string
print(dt.strftime('%m/%d/%y')) # 07/18/20
```

<a name='tz'></a>

## Appendix: See Available Timezones

The common timezone names can be found from pytz

```python
for tz in pytz.common_timezones:
    print(tz)
```


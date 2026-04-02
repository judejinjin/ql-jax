"""Core type definitions, enums, and constants for QL-JAX."""

from enum import IntEnum

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Array = jnp.ndarray

# ---------------------------------------------------------------------------
# Option / Exercise types
# ---------------------------------------------------------------------------

class OptionType(IntEnum):
    Call = 1
    Put = -1


class ExerciseType(IntEnum):
    European = 0
    American = 1
    Bermudan = 2


# ---------------------------------------------------------------------------
# Barrier types
# ---------------------------------------------------------------------------

class BarrierType(IntEnum):
    DownIn = 0
    UpIn = 1
    DownOut = 2
    UpOut = 3


# ---------------------------------------------------------------------------
# Business day conventions
# ---------------------------------------------------------------------------

class BusinessDayConvention(IntEnum):
    Following = 0
    ModifiedFollowing = 1
    Preceding = 2
    ModifiedPreceding = 3
    Unadjusted = 4
    HalfMonthModifiedFollowing = 5
    Nearest = 6


# ---------------------------------------------------------------------------
# Date generation rules
# ---------------------------------------------------------------------------

class DateGeneration(IntEnum):
    Backward = 0
    Forward = 1
    Zero = 2
    ThirdWednesday = 3
    ThirdWednesdayInclusive = 4
    Twentieth = 5
    TwentiethIMM = 6
    OldCDS = 7
    CDS = 8
    CDS2015 = 9


# ---------------------------------------------------------------------------
# Time units / Frequency
# ---------------------------------------------------------------------------

class TimeUnit(IntEnum):
    Days = 0
    Weeks = 1
    Months = 2
    Years = 3
    Hours = 4
    Minutes = 5
    Seconds = 6
    Milliseconds = 7
    Microseconds = 8


class Frequency(IntEnum):
    NoFrequency = -1
    Once = 0
    Annual = 1
    Semiannual = 2
    EveryFourthMonth = 3
    Quarterly = 4
    Bimonthly = 6
    Monthly = 12
    EveryFourthWeek = 13
    Biweekly = 26
    Weekly = 52
    Daily = 365
    OtherFrequency = 999


# ---------------------------------------------------------------------------
# Weekday
# ---------------------------------------------------------------------------

class Month(IntEnum):
    January = 1
    February = 2
    March = 3
    April = 4
    May = 5
    June = 6
    July = 7
    August = 8
    September = 9
    October = 10
    November = 11
    December = 12


class Weekday(IntEnum):
    Sunday = 1
    Monday = 2
    Tuesday = 3
    Wednesday = 4
    Thursday = 5
    Friday = 6
    Saturday = 7


# ---------------------------------------------------------------------------
# Compounding
# ---------------------------------------------------------------------------

class Compounding(IntEnum):
    Simple = 0
    Compounded = 1
    Continuous = 2
    SimpleThenCompounded = 3
    CompoundedThenSimple = 4


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QL_EPSILON = 1.0e-10
QL_NULL_REAL = jnp.finfo(jnp.float64).max
QL_NULL_INTEGER = -1

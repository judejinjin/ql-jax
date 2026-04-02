"""Time basket – allocates a quantity across time buckets."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeBasket:
    """Allocates values to time periods.

    Used in risk engines for theta/time-bucket decomposition of P&L
    and in cashflow analysis for bucketing.

    Parameters
    ----------
    bucket_starts : array – start dates of each bucket
    bucket_ends : array – end dates of each bucket
    """
    bucket_starts: jnp.ndarray
    bucket_ends: jnp.ndarray

    @property
    def n_buckets(self):
        return len(self.bucket_starts)

    def bucket_lengths(self):
        """Length of each bucket."""
        return self.bucket_ends - self.bucket_starts

    def allocate(self, date, amount):
        """Allocate amount to buckets based on date (linear interpolation).

        If date falls between bucket midpoints i and i+1,
        the amount is split proportionally.

        Parameters
        ----------
        date : float
        amount : float

        Returns
        -------
        array – allocated amounts per bucket
        """
        midpoints = 0.5 * (self.bucket_starts + self.bucket_ends)
        n = self.n_buckets
        alloc = jnp.zeros(n)

        # Find which interval the date falls in
        for i in range(n - 1):
            in_range = (date >= midpoints[i]) & (date < midpoints[i + 1])
            w = (date - midpoints[i]) / (midpoints[i + 1] - midpoints[i])
            alloc = jnp.where(
                in_range,
                alloc.at[i].set(amount * (1.0 - w)).at[i + 1].set(amount * w),
                alloc,
            )

        # Before first midpoint
        alloc = jnp.where(date < midpoints[0], alloc.at[0].set(amount), alloc)
        # After last midpoint
        alloc = jnp.where(date >= midpoints[-1], alloc.at[-1].set(amount), alloc)

        return alloc

    def allocate_cashflows(self, dates, amounts):
        """Allocate multiple cashflows into buckets.

        Parameters
        ----------
        dates : array
        amounts : array

        Returns
        -------
        array – total allocated amount per bucket
        """
        total = jnp.zeros(self.n_buckets)
        for i in range(len(dates)):
            total = total + self.allocate(dates[i], amounts[i])
        return total

import asyncio
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Dict, Optional, Type

from aiohttp import ClientSession


class AsyncRateLimiter(AbstractAsyncContextManager):
    def __init__(self, max_acquisitions: float, time_period: float = 60, burst: bool = False) -> None:
        """
        A leaky bucket rate limiter.
        This is an asynchronous context manager when used with :keyword:`async with`, entering the
        context acquires capacity:
            limiter = AsyncRateLimiter(10)
            for foo in bar:
                async with limiter:
                    # process foo elements at 10 items per minute
        This code has been repurposed from https://github.com/mjpieters/aiolimiter.
        
        Parameters
        ----------
        max_acquisitions : float
            Allow up to `max_acquisitions` / `time_period` acquisitions before blocking.
        time_period : float
            duration, in seconds, of the time period in which to limit the rate. Note that up to
            `max_acquisitions` acquisitions are allowed within this time period in a burst, unless burst is
            set to False.
        burst : bool
            The ability to burst the max capacity first, before limiting the rate
        """
        assert max_acquisitions > 0

        if burst:
            self.max_acquisitions = max_acquisitions
            self.time_period = time_period
        else:
            # If burst is False, limit the rate to the rate limit from the offset
            self.max_acquisitions = 1
            self.time_period = time_period / max_acquisitions
        # Set internal variables
        self._acquisitions_per_sec = max_acquisitions / time_period
        self._level = 0.0
        self._last_check = 0.0
        # queue of waiting futures to signal capacity to
        self._waiters: Dict[asyncio.Task, asyncio.Future] = {}

    def _leak(self) -> None:
        """Drip out capacity from the bucket."""
        loop = asyncio.get_running_loop()
        if self._level:
            # drip out enough level for the elapsed time since
            # we last checked
            elapsed = loop.time() - self._last_check
            decrement = elapsed * self._acquisitions_per_sec
            self._level = max(self._level - decrement, 0)
        self._last_check = loop.time()

    def has_capacity(self, amount: float = 1) -> bool:
        """
        Check if there is enough capacity remaining in the limiter
        
        Parameters
        ----------
        amount : float
            How much capacity you need to be available.
        """
        self._leak()
        requested = self._level + amount
        # if there are tasks waiting for capacity, signal to the first
        # there there may be some now (they won't wake up until this task
        # yields with an await)
        if requested < self.max_acquisitions:
            for fut in self._waiters.values():
                if not fut.done():
                    fut.set_result(True)
                    break
        return self._level + amount <= self.max_acquisitions

    async def acquire(self, amount: float = 1) -> None:
        """
        Acquire capacity in the limiter.
        If the limit has been reached, blocks until enough capacity has been
        freed before returning.
        
        Parameters
        ----------
        amount : float
            How much capacity you need to be available.
        """
        if amount > self.max_acquisitions:
            raise ValueError("Can't acquire more than the maximum capacity")

        loop = asyncio.get_running_loop()
        task = asyncio.current_task(loop)
        assert task is not None
        while not self.has_capacity(amount):
            # wait for the next drip to have left the bucket
            # add a future to the _waiters map to be notified
            # 'early' if capacity has come up
            fut = loop.create_future()
            self._waiters[task] = fut
            try:
                await asyncio.wait_for(asyncio.shield(fut), 1 / self._acquisitions_per_sec * amount)
            except asyncio.TimeoutError:
                pass
            fut.cancel()
        self._waiters.pop(task, None)

        self._level += amount

        return None

    async def __aenter__(self) -> None:
        await self.acquire()
        return None

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        return None


class LimitedClientSession(ClientSession):
    def __init__(
        self, requests_per_second: Optional[float] = None, burst: bool = False, *args, **kwargs
    ) -> None:
        """
        Throttles methods.
        
        Parameters
        ----------
        requests_per_second : Optional[float]
            Maximum number of requests per second to allow. If None, no rate
            limiting is applied and the resulting behaviour is the same as
            a standard ClientSession. However, in these scenarios, using a
            standard ClientSession is recommended for performance reasons.
        *args
            Any positional arguments to initialise the underlying ClientSession with.
        **kwargs
            Any keyword arguments to initialise the underlying ClientSession with.
        """
        super().__init__(*args, **kwargs)
        if requests_per_second:
            self.rate_limiter = AsyncRateLimiter(
                max_acquisitions=requests_per_second, time_period=1, burst=burst
            )
        else:
            self.rate_limiter = None

    async def get(self, *args, **kwargs):
        """
        Call the client's get method, at the rate of our rate limiter.
        """
        if not self.rate_limiter:
            return await super().get(*args, **kwargs)

        async with self.rate_limiter:
            return await super().get(*args, **kwargs)

    async def post(self, *args, **kwargs):
        """
        Call the client's post method, at the rate of our rate limiter.
        """
        if not self.rate_limiter:
            return await super().post(*args, **kwargs)

        async with self.rate_limiter:
            return await super().post(*args, **kwargs)

    async def request(self, *args, **kwargs):
        """
        Call the client's request method, at the rate of our rate limiter.
        """
        if not self.rate_limiter:
            return await super().request(*args, **kwargs)

        async with self.rate_limiter:
            return await super().request(*args, **kwargs)
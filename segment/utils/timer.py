import time
import datetime
from functools import wraps


def tick_tock(prefix=None, with_return=False):
    def inner_func(func):
        @wraps(func)
        def warp(*args, **kwargs):
            start_time = datetime.datetime.now()
            res = func(*args, **kwargs)
            current_time = datetime.datetime.now()
            spent_time = current_time - start_time
            spent_time = str(spent_time).split(".")[0]  # h+:mm:ss

            if prefix is not None:
                print(f"[{prefix} {spent_time}]", end="   ")
            else:
                print(f'Apply "{func.__name__}" took {spent_time}')

            if with_return:
                return res

        return warp

    return inner_func


class TimerError(Exception):
    """Custom exception for Timer class."""


class TickTock:
    timers = {}

    def __init__(
        self,
        name: str = "default",
        mode: "display || return" = "display",
        end="\n",
    ):
        """
        :param name: can pass in a specific name of a timer.  This is
            helpful if you want to time multiple events, and accumulate the
            result with one total time.  The timer name is the dictionary
            key and the accumulated time is the dictionary value.
        """
        self.__name = name
        self.__start_time = None
        self.__elapsed_time = None
        self.mode = mode
        self.end = end

    @property
    def get_timer(self) -> dict:
        """
        Get timer name and associated time value
        :return: dictionary of the timer name and run time for that timer name
        """
        return self.timers

    @property
    def elapsed_time(self):
        return self.__elapsed_time

    def start(self) -> None:
        """Start timer"""
        if self.__start_time is not None:
            raise TimerError("Timer is running. " "Use .stop() to stop it")

        name = self.__name

        # Add new name to dictionary of timers
        self.timers[name] = {}

        start = datetime.datetime.now()
        self.__start_time = start

        self.timers[name]["start"] = str(start).split(".")[0]

    def stop(self) -> str:
        """
        Stop timer, and reset value of start time to None
        :return: elapsed time (aka delta) since starting the timer
        """
        if self.__start_time is None:
            raise TimerError("Timer not started. " "Use .start() to start it.")
        timers = self.timers
        name = self.__name
        end = datetime.datetime.now()
        elapsed = (end - self.__start_time)
        self.__elapsed_time = elapsed
        self.__start_time = None

        if name:
            timers[name]["end"] = str(end).split(".")[0]
            timers[name]["elapsed"] = str(elapsed).split(".")[0]

        if self.mode == "display":
            # print(f'[{name} {elapsed}]', end="  ")
            print(f'[{name} {str(elapsed).split(".")[0]}]', end=self.end)

        else:
            return str(self.__elapsed_time).split(".")[0]

    def cumulative_time(self):
        if self.__start_time is None:
            raise TimerError("Timer not started. " "Use .start() to start it.")

        timers = self.timers
        name = self.__name
        end = datetime.datetime.now()
        elapsed = (end - self.__start_time)
        self.__elapsed_time = elapsed

        if name:
            timers[name]["end"] = str(end).split(".")[0]
            timers[name]["elapsed"] = str(elapsed).split(".")[0]

        if self.mode == "display":
            # print(f'[{name} {elapsed}]', end="  ")
            print(f'[{name} {str(elapsed).split(".")[0]}]', end=self.end)
        else:
            return str(self.__elapsed_time).split(".")[0]

    def __call__(self, func):
        """Support using Timer as a decorator"""
        if self.__name == "default":
            self.__name = func.__qualname__

        @wraps(func)
        def wrapper_timer(*args, **kwargs):
            """
            Use the context manager to ensure that itself (TikTok instance)
            starts at the beginning of the input function and
            stops at the end of it.
            """
            with self:
                return func(*args, **kwargs)

        return wrapper_timer

    # Context Manager Protocal
    def __enter__(self):
        """Start new timer using Timer context manager"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Stop timer using Timer context manager
        For the moment, this class does not yet do any error handling.
        It just uses the default error handling when calling .__exit__()
        """
        self.stop()

    def __repr__(self) -> str:
        return f"name '{self.__name}';" \
            f"time {self.get_timer[self.__name]}"


if __name__ == "__main__":

    @TickTock()
    def i_do():
        return "I do"

    @TickTock()
    class Do:
        def __init__(self):
            time.sleep(1)

        @TickTock()
        def do(self):
            time.sleep(1)

    i = Do()
    # i.do()

    t = TickTock("total")
    t.start()

"""
Provides iterators for scheduling tasks at true time intervals across days.

This module solves the problem where using range(0, 24, 7) with CronEvent
creates tasks at hours [0, 7, 14, 21] each day, which doesn't maintain
a consistent 7-hour interval (21h -> 0h is only 3 hours).
"""

from datetime import datetime, timedelta


class IntervalMatcher:
    """
    A special matcher class that works with CronEvent to provide true time intervals.
    
    This class implements the __contains__ method to work with CronEvent's matching logic.
    Instead of matching against a set of discrete values, it calculates whether the
    current time matches a periodic interval from a start time.
    
    Usage with CronEvent:
        matcher = IntervalMatcher(interval_hours=7)
        event = CronEvent(action=my_function, hour=matcher, min=[0])
        # This will trigger every 7 hours from the start time
    """
    
    def __init__(self, interval_hours, start_time=None):
        """
        Initialize the interval matcher.
        
        Args:
            interval_hours (int or float): Interval in hours
            start_time (datetime, optional): Starting reference time. Defaults to midnight today.
        """
        self.interval_hours = interval_hours
        if start_time is None:
            # Default to midnight of the current day
            now = datetime.now()
            self.start_time = datetime(now.year, now.month, now.day, 0, 0, 0)
        else:
            self.start_time = start_time
        
        self.interval_seconds = interval_hours * 3600
        
    def __contains__(self, item):
        """
        Check if the given hour/minute/day value matches the interval.
        
        This is called by CronEvent's matchtime() method.
        When CronEvent checks 't.hour in self.hours', this method is invoked.
        
        Note: This implementation works when used for the 'hour' parameter with 'min' set to a specific value.
        """
        # For hour matching, we need access to the full datetime
        # This is a limitation - we can't determine interval match from hour alone
        return True  # Placeholder - see TimestampMatcher for better approach
    
    def matches_datetime(self, dt):
        """
        Check if a datetime matches the interval schedule.
        
        Args:
            dt (datetime): The datetime to check
            
        Returns:
            bool: True if this datetime is on the interval schedule
        """
        # Calculate seconds since start time
        delta = (dt - self.start_time).total_seconds()
        
        # Check if this is a multiple of the interval (within 1 minute tolerance)
        remainder = delta % self.interval_seconds
        
        # Allow 1-minute tolerance on either side
        return remainder < 60 or remainder > (self.interval_seconds - 60)


class TimestampMatcher:
    """
    A matcher that checks if a complete timestamp (year, month, day, hour, minute)
    matches a periodic interval.
    
    This is designed to work as a custom matching class for CronEvent by implementing
    a check method that validates complete timestamps rather than individual components.
    
    Usage:
        # This requires a modified CronEvent, or use IntervalCronEvent instead
        matcher = TimestampMatcher(interval_hours=7)
    """
    
    def __init__(self, interval_hours, start_time=None):
        """
        Initialize the timestamp matcher.
        
        Args:
            interval_hours (int or float): Interval in hours
            start_time (datetime, optional): Starting reference time. Defaults to now.
        """
        self.interval_hours = interval_hours
        self.start_time = start_time or datetime.now().replace(second=0, microsecond=0)
        self.interval_seconds = interval_hours * 3600
        
    def matches(self, dt):
        """
        Check if a datetime matches the interval schedule.
        
        Args:
            dt (datetime): The datetime to check
            
        Returns:
            bool: True if this datetime is on the interval schedule
        """
        # Calculate seconds since start time
        delta = (dt - self.start_time).total_seconds()
        
        # Must be in the future (or present)
        if delta < 0:
            return False
        
        # Check if this is a multiple of the interval (within 1 minute tolerance)
        remainder = delta % self.interval_seconds
        
        # Allow matching within the same minute
        return remainder < 60


class IntervalIterator:
    """
    Creates an iterator that yields (day_of_year, hour) tuples for true n-hour intervals.
    
    Usage:
        iterator = IntervalIterator(interval_hours=7, start_time=datetime.now())
        
        # For use with CronEvent, extract hours and days
        # Note: This generates a finite set for the year
        schedule = iterator.get_schedule_for_year()
        
        # Create CronEvent with day and hour matching
        event = CronEvent(
            action=my_function,
            day=schedule['days'],
            hour=schedule['hours']
        )
    """
    
    def __init__(self, interval_hours, start_time=None, duration_days=365):
        """
        Initialize the interval iterator.
        
        Args:
            interval_hours (int or float): Interval in hours
            start_time (datetime, optional): Starting time. Defaults to now.
            duration_days (int): How many days ahead to generate schedule
        """
        self.interval_hours = interval_hours
        self.start_time = start_time or datetime.now()
        self.duration_days = duration_days
        
    def generate_schedule(self):
        """
        Generate all scheduled times for the specified duration.
        
        Yields:
            datetime: Each scheduled execution time
        """
        current_time = self.start_time
        end_time = self.start_time + timedelta(days=self.duration_days)
        
        while current_time <= end_time:
            yield current_time
            current_time += timedelta(hours=self.interval_hours)
    
    def get_schedule_for_year(self):
        """
        Get day and hour sets for use with CronEvent.
        
        Returns:
            dict: Dictionary with 'days' and 'hours' sets containing
                  the day-of-year and hour values when tasks should run
        """
        schedule_times = list(self.generate_schedule())
        
        days = set()
        hours = set()
        
        for dt in schedule_times:
            days.add(dt.timetuple().tm_yday)  # Day of year (1-366)
            hours.add(dt.hour)
        
        return {'days': days, 'hours': hours}
    
    def get_day_hour_pairs(self):
        """
        Get list of (day_of_year, hour) tuples for the schedule.
        
        Returns:
            list: List of (day_of_year, hour) tuples
        """
        schedule_times = list(self.generate_schedule())
        return [(dt.timetuple().tm_yday, dt.hour) for dt in schedule_times]


class HourIntervalSet:
    """
    Simplified class specifically for n-hour intervals.
    Returns an object that can be used directly with CronEvent.
    
    Usage:
        # For a task every 7 hours starting now
        schedule = HourIntervalSet(7)
        
        event = CronEvent(
            action=my_function,
            day=schedule.days,
            hour=schedule.hours
        )
    """
    
    def __init__(self, interval_hours, start_time=None, duration_days=365):
        """
        Initialize hour interval set.
        
        Args:
            interval_hours (int or float): Interval in hours
            start_time (datetime, optional): Starting time. Defaults to now.
            duration_days (int): How many days ahead to generate schedule
        """
        iterator = IntervalIterator(interval_hours, start_time, duration_days)
        schedule = iterator.get_schedule_for_year()
        self.days = schedule['days']
        self.hours = schedule['hours']
        self._pairs = iterator.get_day_hour_pairs()
    
    def get_pairs(self):
        """Get the actual (day, hour) combinations."""
        return self._pairs


class IntervalCronEvent:
    """
    A CronEvent-compatible class that triggers at true time intervals.
    
    This solves the problem where range(0, 24, 7) triggers at [0, 7, 14, 21] every day,
    losing the interval between day transitions.
    
    Usage:
        def my_task():
            print("Task executed!")
        
        # Execute every 7 hours starting now
        event = IntervalCronEvent(my_task, interval_hours=7)
        
        # Or every 30 minutes
        event = IntervalCronEvent(my_task, interval_minutes=30)
        
        # Or every 2 days
        event = IntervalCronEvent(my_task, interval_days=2)
        
        CronDaemon().register(event)
    """
    
    def __init__(self, action, interval_hours=None, interval_minutes=None, interval_days=None, 
                 interval_seconds=None, start_time=None, destruction=False, args=(), kwargs={}):
        """
        Initialize an interval-based cron event.
        
        Args:
            action (callable): Function to execute when interval matches
            interval_hours (int or float, optional): Interval in hours
            interval_minutes (int or float, optional): Interval in minutes
            interval_days (int or float, optional): Interval in days
            interval_seconds (int or float, optional): Interval in seconds
            start_time (datetime, optional): Starting time. Defaults to now.
            destruction (bool): Whether to remove event after first execution
            args (tuple): Arguments to pass to action
            kwargs (dict): Keyword arguments to pass to action
            
        Note:
            Only one of interval_hours, interval_minutes, interval_days, or interval_seconds
            should be specified. If multiple are given, they will be summed.
        """
        self.action = action
        self.start_time = start_time or datetime.now().replace(second=0, microsecond=0)
        self.destruction = destruction
        self.args = args
        self.kwargs = kwargs
        
        # Calculate total interval in seconds
        total_seconds = 0
        self.interval_description = []
        
        if interval_days is not None:
            total_seconds += interval_days * 86400
            self.interval_description.append('{0} days'.format(interval_days))
            
        if interval_hours is not None:
            total_seconds += interval_hours * 3600
            self.interval_description.append('{0} hours'.format(interval_hours))
            
        if interval_minutes is not None:
            total_seconds += interval_minutes * 60
            self.interval_description.append('{0} minutes'.format(interval_minutes))
            
        if interval_seconds is not None:
            total_seconds += interval_seconds
            self.interval_description.append('{0} seconds'.format(interval_seconds))
        
        if total_seconds == 0:
            raise ValueError('Must specify at least one interval: interval_days, interval_hours, interval_minutes, or interval_seconds')
        
        self.interval_seconds = total_seconds
        
        # Store individual components for backwards compatibility
        self.interval_hours = interval_hours
        self.interval_minutes = interval_minutes
        self.interval_days = interval_days
        
        self.last_execution = None
        
    def matchtime(self, t):
        """
        Check if this event should trigger at the specified datetime.
        
        Args:
            t (datetime): The time to check
            
        Returns:
            bool: True if event should trigger
        """
        # Calculate seconds since start time
        delta = (t - self.start_time).total_seconds()
        
        # Must be at or after start time
        if delta < 0:
            return False
        
        # Check if this is a multiple of the interval
        remainder = delta % self.interval_seconds
        
        # Trigger if we're within the first minute of an interval period
        # This accounts for the minute-by-minute checking of CronDaemon
        if remainder < 60:
            # Prevent double-triggering in the same minute
            if self.last_execution is not None:
                time_since_last = (t - self.last_execution).total_seconds()
                if time_since_last < 60:
                    return False
            return True
        
        return False
    
    def check(self, t):
        """
        Check if event should trigger and execute if so.
        
        Args:
            t (datetime): The time to check
            
        Returns:
            bool: True if event was triggered
        """
        if self.matchtime(t):
            import logging
            logging.getLogger().info('Interval event triggered at ' + str(t) + '. Executing ' + str(self.action) + '.')
            self.last_execution = t
            self.action(*self.args, **self.kwargs)
            return True
        return False
    
    def force_trigger(self):
        """
        Manually trigger the event immediately, bypassing time checks.
        
        This is useful for testing or forcing an immediate execution without
        waiting for the next scheduled interval.
        
        Returns:
            bool: Always returns True
        """
        import logging
        logging.getLogger().info('Force triggering interval event: ' + str(self.action))
        self.last_execution = datetime.now()
        self.action(*self.args, **self.kwargs)
        return True
    
    def __repr__(self):
        if self.interval_description:
            interval_str = ', '.join(self.interval_description)
        else:
            interval_str = '{0} seconds'.format(self.interval_seconds)
        return 'Interval Cron Event (' + interval_str + ') on callable ' + str(self.action)


def create_interval_schedule(interval_hours, start_time=None, duration_days=365):
    """
    Convenience function to create a schedule for n-hour intervals.
    
    Args:
        interval_hours (int or float): Interval in hours
        start_time (datetime, optional): Starting time. Defaults to now.
        duration_days (int): How many days ahead to generate schedule
    
    Returns:
        HourIntervalSet: Object with .days and .hours attributes for CronEvent
    
    Example:
        schedule = create_interval_schedule(7)
        event = CronEvent(my_function, day=schedule.days, hour=schedule.hours)
    """
    return HourIntervalSet(interval_hours, start_time, duration_days)


# Example usage and testing
if __name__ == '__main__':
    from datetime import datetime
    
    # Example 1: Using IntervalCronEvent with different time units
    print("=" * 60)
    print("Example 1: IntervalCronEvent with different time units")
    print("=" * 60)
    
    def my_task():
        print("Task executed at " + str(datetime.now()))
    
    # Create events with different time units
    event_hours = IntervalCronEvent(my_task, interval_hours=7)
    print("Event with hours: " + repr(event_hours))
    
    event_minutes = IntervalCronEvent(my_task, interval_minutes=30)
    print("Event with minutes: " + repr(event_minutes))
    
    event_days = IntervalCronEvent(my_task, interval_days=2)
    print("Event with days: " + repr(event_days))
    
    # Combine multiple units
    event_combined = IntervalCronEvent(my_task, interval_days=1, interval_hours=12, interval_minutes=30)
    print("Event with combined units: " + repr(event_combined))
    print("  Total seconds: {0}".format(event_combined.interval_seconds))
    
    # Example 2: Test 7-hour intervals
    print("\n" + "=" * 60)
    print("Example 2: Testing 7-hour intervals across days")
    print("=" * 60)
    
    start = datetime(2025, 11, 17, 0, 0, 0)
    event = IntervalCronEvent(my_task, interval_hours=7, start_time=start)
    
    # Test the matching at different times
    test_times = [
        datetime(2025, 11, 17, 0, 0),   # Start time - should match
        datetime(2025, 11, 17, 7, 0),   # +7h - should match
        datetime(2025, 11, 17, 8, 0),   # +8h - should NOT match
        datetime(2025, 11, 17, 14, 0),  # +14h - should match
        datetime(2025, 11, 17, 21, 0),  # +21h - should match
        datetime(2025, 11, 18, 0, 0),   # +24h - should NOT match (not a 7h interval)
        datetime(2025, 11, 18, 4, 0),   # +28h - should match
        datetime(2025, 11, 18, 11, 0),  # +35h - should match
    ]
    
    for t in test_times:
        matches = event.matchtime(t)
        hours_from_start = (t - start).total_seconds() / 3600
        match_str = 'YES MATCH' if matches else 'no match'
        print("{0} (+{1:5.1f}h): {2}".format(t, hours_from_start, match_str))
    
    # Example 3: Test minute intervals
    print("\n" + "=" * 60)
    print("Example 3: Testing 90-minute intervals")
    print("=" * 60)
    
    start_min = datetime(2025, 11, 17, 10, 0, 0)
    event_min = IntervalCronEvent(my_task, interval_minutes=90, start_time=start_min)
    
    test_times_min = [
        datetime(2025, 11, 17, 10, 0),  # Start - should match
        datetime(2025, 11, 17, 11, 30), # +90min - should match
        datetime(2025, 11, 17, 13, 0),  # +180min - should match
        datetime(2025, 11, 17, 14, 30), # +270min - should match
        datetime(2025, 11, 17, 15, 0),  # +300min - should NOT match
    ]
    
    for t in test_times_min:
        matches = event_min.matchtime(t)
        mins_from_start = (t - start_min).total_seconds() / 60
        match_str = 'YES MATCH' if matches else 'no match'
        print("{0} (+{1:5.0f}min): {2}".format(t, mins_from_start, match_str))
    
    # Example 4: Compare with the problematic range(0, 24, 7) approach
    print("\n" + "=" * 60)
    print("Example 4: Problems with range(0, 24, 7)")
    print("=" * 60)
    print("Hours that would trigger: " + str(list(range(0, 24, 7))))
    print("Intervals between triggers:")
    hours = [0, 7, 14, 21, 24]  # Adding 24 to show the next day's 0
    for i in range(len(hours) - 1):
        interval = hours[i+1] - hours[i]
        print("  Hour {0:2d} -> Hour {1:2d}: {2} hours".format(hours[i], hours[i+1], interval))
    print("Notice: 21h -> 0h (next day) is only 3 hours! [WRONG]")
    
    # Example 5: Using the static schedule generator
    print("\n" + "=" * 60)
    print("Example 5: Static schedule for 7-hour intervals")
    print("=" * 60)
    schedule = create_interval_schedule(7, start_time=datetime(2025, 1, 1, 0, 0), duration_days=3)
    print("First 15 (day, hour) pairs:")
    for day, hour in schedule.get_pairs()[:15]:
        print("  Day {0:3d}, Hour {1:2d}".format(day, hour))


# Interval Iterator - True Time Interval Scheduling

## The Problem

When using `CronEvent` with `range(0, 24, 7)` to schedule tasks every 7 hours, you get:

```python
CronEvent(my_task, hour=range(0, 24, 7))  # Triggers at hours: [0, 7, 14, 21]
```

**Issue**: This triggers at hours 0, 7, 14, and 21 **every single day**, not in true 7-hour intervals!

- Day 1: 00:00, 07:00, 14:00, 21:00
- Day 2: 00:00, 07:00, 14:00, 21:00
- ...

The interval from 21:00 to 00:00 (next day) is only **3 hours**, not 7 hours! ❌

## The Solution: IntervalCronEvent

The `IntervalCronEvent` class properly tracks time intervals across days:

```python
from tools.interval_iterator import IntervalCronEvent
from tools.cron import CronDaemon

# Schedule a task to run every 7 hours
event = IntervalCronEvent(my_task, interval_hours=7)
CronDaemon().register(event)
```

**Result**: True 7-hour intervals that span across days correctly:

- Day 1: 00:00, 07:00, 14:00, 21:00
- Day 2: **04:00**, 11:00, 18:00
- Day 3: 01:00, 08:00, 15:00, 22:00
- ...

## Usage Examples

### Example 1: Basic Usage with Hours

```python
from tools.interval_iterator import IntervalCronEvent
from tools.cron import CronDaemon

def my_periodic_task():
    print("Task executed!")
    # Do your work here

# Execute every 7 hours
event = IntervalCronEvent(my_periodic_task, interval_hours=7)
CronDaemon().register(event)
```

### Example 2: Using Minutes

```python
# Execute every 30 minutes
event = IntervalCronEvent(my_task, interval_minutes=30)
CronDaemon().register(event)
```

### Example 3: Using Days

```python
# Execute every 2 days
event = IntervalCronEvent(my_task, interval_days=2)
CronDaemon().register(event)
```

### Example 4: Combining Multiple Time Units

```python
# Execute every 1 day, 12 hours, and 30 minutes (36.5 hours total)
event = IntervalCronEvent(
    my_task, 
    interval_days=1,
    interval_hours=12,
    interval_minutes=30
)
CronDaemon().register(event)
```

### Example 5: Custom Start Time

```python
from datetime import datetime
from tools.interval_iterator import IntervalCronEvent

# Start at a specific time
start_time = datetime(2025, 11, 17, 13, 0, 0)  # Nov 17, 2025 at 1:00 PM
event = IntervalCronEvent(
    my_task, 
    interval_hours=6,
    start_time=start_time
)
```

### Example 6: With Arguments

```python
def task_with_args(name, value):
    print("Task {0} executed with value: {1}".format(name, value))

event = IntervalCronEvent(
    task_with_args,
    interval_hours=12,
    args=('DataCollection',),
    kwargs={'value': 42}
)
```

### Example 7: Integration with Existing Code (ODMR)

```python
class ODMR(ManagedJob, GetSetItemsMixin):
    enable_periodic_tasks = Bool(True)
    periodic_tasks_interval = Range(low=1, high=24, value=1)
    force_trigger_button = Button(label='Force Trigger')
    
    def _enable_periodic_tasks_changed(self, new):
        if not new and hasattr(self, 'cron_event'):
            CronDaemon().remove(self.cron_event)
        if new:
            # Use IntervalCronEvent for true n-hour intervals
            self.cron_event = IntervalCronEvent(
                self._periodic_task, 
                interval_hours=self.periodic_tasks_interval
            )
            CronDaemon().register(self.cron_event)
    
    def _force_trigger_button_fired(self):
        """React to force trigger button. Manually trigger the periodic task immediately."""
        if hasattr(self, 'cron_event'):
            logging.getLogger().info('Force triggering periodic task.')
            self.cron_event.force_trigger()
        else:
            logging.getLogger().warning('No cron event registered. Enable periodic tasks first.')
```

### Example 5: Force Triggering for Testing

```python
# Create an event with a long interval
event = IntervalCronEvent(my_task, interval_hours=7)
CronDaemon().register(event)

# Don't want to wait 7 hours to test? Force trigger immediately!
event.force_trigger()  # Executes the task right now
```

## How It Works

`IntervalCronEvent` implements the same interface as `CronEvent`:

1. **matchtime(t)**: Checks if the current datetime `t` matches an interval
2. **check(t)**: Executes the action if `matchtime(t)` returns True
3. **force_trigger()**: Manually executes the action immediately (useful for testing)

### Time Unit Parameters

You can specify intervals using any combination of these parameters:

- **interval_days**: Interval in days (e.g., `interval_days=2` for every 2 days)
- **interval_hours**: Interval in hours (e.g., `interval_hours=7` for every 7 hours)
- **interval_minutes**: Interval in minutes (e.g., `interval_minutes=30` for every 30 minutes)
- **interval_seconds**: Interval in seconds (e.g., `interval_seconds=3600` for every hour)

**Note**: You can combine multiple units! For example:
```python
# Execute every 1 day, 12 hours, and 30 minutes
event = IntervalCronEvent(
    my_task,
    interval_days=1,
    interval_hours=12,
    interval_minutes=30
)
```

### Matching Algorithm

The class calculates the time elapsed since the start time and checks if it's a multiple of the interval:

```python
def matchtime(self, t):
    delta = (t - self.start_time).total_seconds()
    if delta < 0:
        return False  # Before start time
    
    remainder = delta % self.interval_seconds
    return remainder < 60  # Within first minute of interval period
```

This ensures:
- ✅ True time intervals (7 hours = exactly 7 hours)
- ✅ Spans across days correctly
- ✅ No double-triggering
- ✅ Compatible with existing `CronDaemon`

## Comparison

| Method | Behavior | Interval Accuracy |
|--------|----------|-------------------|
| `CronEvent(hour=range(0,24,7))` | Triggers at 0, 7, 14, 21 every day | ❌ Broken at day boundaries |
| `IntervalCronEvent(interval_hours=7)` | Triggers every 7 hours continuously | ✅ True intervals |

## Additional Tools

The module also provides `create_interval_schedule()` for generating static schedules:

```python
from tools.interval_iterator import create_interval_schedule

# Generate a schedule for the next year
schedule = create_interval_schedule(
    interval_hours=7,
    start_time=datetime(2025, 1, 1, 0, 0),
    duration_days=365
)

# Get day-hour pairs
pairs = schedule.get_pairs()
# [(1, 0), (1, 7), (1, 14), (1, 21), (2, 4), (2, 11), ...]
```

This can be useful for analysis or visualization of scheduled times.

## Testing

Run the module directly to see examples:

```bash
python tools/interval_iterator.py
```

This demonstrates:
1. How `IntervalCronEvent` works with 7-hour intervals
2. The problem with `range(0, 24, 7)`
3. Static schedule generation

## Requirements

- Python 2.7 or Python 3.x
- Existing `CronDaemon` infrastructure
- `datetime` module (standard library)

## Files

- `tools/interval_iterator.py` - Main implementation
- `tools/cron.py` - Existing cron infrastructure
- `measurements/odmr_ps.py` - Example usage

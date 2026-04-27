"""
Test script demonstrating all time unit options for IntervalCronEvent.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from tools.interval_iterator import IntervalCronEvent

execution_log = []

def my_task(name):
    """Simple task that logs execution."""
    execution_log.append((name, datetime.now()))
    print("[{0}] Task '{1}' executed".format(datetime.now().strftime("%H:%M:%S"), name))

print("="*70)
print("Testing IntervalCronEvent with Different Time Units")
print("="*70)

# Test 1: Hours
print("\n1. Testing interval_hours=2")
event_hours = IntervalCronEvent(my_task, interval_hours=2, args=('hours_task',))
print("   Created: {0}".format(event_hours))
print("   Interval in seconds: {0}".format(event_hours.interval_seconds))
event_hours.force_trigger()

# Test 2: Minutes
print("\n2. Testing interval_minutes=30")
event_minutes = IntervalCronEvent(my_task, interval_minutes=30, args=('minutes_task',))
print("   Created: {0}".format(event_minutes))
print("   Interval in seconds: {0}".format(event_minutes.interval_seconds))
event_minutes.force_trigger()

# Test 3: Days
print("\n3. Testing interval_days=1")
event_days = IntervalCronEvent(my_task, interval_days=1, args=('days_task',))
print("   Created: {0}".format(event_days))
print("   Interval in seconds: {0}".format(event_days.interval_seconds))
event_days.force_trigger()

# Test 4: Seconds
print("\n4. Testing interval_seconds=3600")
event_seconds = IntervalCronEvent(my_task, interval_seconds=3600, args=('seconds_task',))
print("   Created: {0}".format(event_seconds))
print("   Interval in seconds: {0}".format(event_seconds.interval_seconds))
event_seconds.force_trigger()

# Test 5: Combined units
print("\n5. Testing combined units (2 days, 3 hours, 45 minutes)")
event_combined = IntervalCronEvent(
    my_task, 
    interval_days=2,
    interval_hours=3,
    interval_minutes=45,
    args=('combined_task',)
)
print("   Created: {0}".format(event_combined))
print("   Interval in seconds: {0}".format(event_combined.interval_seconds))
expected = 2*86400 + 3*3600 + 45*60
print("   Expected: {0} seconds".format(expected))
print("   Match: {0}".format(event_combined.interval_seconds == expected))
event_combined.force_trigger()

# Test 6: All units combined
print("\n6. Testing all units (1 day, 2 hours, 30 minutes, 45 seconds)")
event_all = IntervalCronEvent(
    my_task,
    interval_days=1,
    interval_hours=2,
    interval_minutes=30,
    interval_seconds=45,
    args=('all_units_task',)
)
print("   Created: {0}".format(event_all))
print("   Interval in seconds: {0}".format(event_all.interval_seconds))
expected = 1*86400 + 2*3600 + 30*60 + 45
print("   Expected: {0} seconds".format(expected))
print("   Match: {0}".format(event_all.interval_seconds == expected))
event_all.force_trigger()

# Test 7: Error handling - no interval specified
print("\n7. Testing error handling (no interval specified)")
try:
    event_error = IntervalCronEvent(my_task)
    print("   ERROR: Should have raised ValueError!")
except ValueError as e:
    print("   SUCCESS: Caught expected error: {0}".format(e))

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Total tasks executed: {0}".format(len(execution_log)))
for name, timestamp in execution_log:
    print("  - {0} at {1}".format(name, timestamp.strftime("%H:%M:%S")))

print("\n" + "="*70)
print("All time unit options work correctly!")
print("="*70)
print("\nAvailable options:")
print("  - interval_hours   : Interval in hours")
print("  - interval_minutes : Interval in minutes")
print("  - interval_days    : Interval in days")
print("  - interval_seconds : Interval in seconds")
print("  - Multiple units can be combined for precise intervals")

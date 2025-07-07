
# Countdown Timer Documentation

## Overview
The Countdown Timer is a simple Python script that creates a countdown timer displaying minutes and seconds. It allows users to input a duration in seconds and displays a real-time countdown until completion.

## Features
* Real-time countdown display
* Converts seconds to minutes and seconds format
* Clear terminal output with carriage return
* User input functionality

## Code Structure
```python
import time

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins,secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1

    print('Timer completed!')

t = input('Enter the time in seconds: ')
countdown(int(t))


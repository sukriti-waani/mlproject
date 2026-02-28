# Importing logging module
# logging is used to record events, errors, warnings during program execution
import logging

# Importing os module
# os helps in interacting with operating system (folders, paths, files)
import os

# Importing datetime class
# Used to get current date and time
from datetime import datetime


# Creating a log file name dynamically using current date & time
# strftime() formats the date into Month_Day_Year_Hour_Minute_Second
# Example: 09_01_2026_21_45_30.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


# Creating path for logs directory
# os.getcwd() → returns current working directory
# "logs" → folder name
# os.path.join() safely joins paths (works in Windows/Mac/Linux)
logs_dir = os.path.join(os.getcwd(), "logs")


# Creating logs folder if it does not exist
# exist_ok=True → prevents error if folder already exists
os.makedirs(logs_dir, exist_ok=True)


# Creating full path of log file
# Combines logs folder path with file name
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)


# Configuring logging settings
logging.basicConfig(

    # filename → where logs will be stored
    filename=LOG_FILE_PATH,

    # format → defines how log message will appear
    # %(asctime)s → time of log
    # %(lineno)d → line number where log was called
    # %(name)s → module name
    # %(levelname)s → INFO / ERROR / WARNING
    # %(message)s → actual log message
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",

    # level → minimum level of logs to record
    # INFO means INFO, WARNING, ERROR, CRITICAL will be recorded
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")



# What is Logger?
# A logger is used to record what happens inside your program.


# It stores messages like:
    # When the program started
    # When an error occurred
    # Which line caused the error


# Why Do We Use Logger?
    # Because print() only shows output on screen.
    # Logger saves everything in a file for future checking.


# Logger Setup (What It Does)
# code:
    # Creates a logs folder
    # Creates a log file with current date & time
    # Stores logs inside that file
    # Saves details like time, line number, log level, and message
# Importing sys module
# sys module helps us access system-specific parameters and functions
# We use it here to get detailed exception information
import sys
from src.logger import logging

# This function creates a detailed error message
# error → actual error object/message
# error_detail → system module used to extract traceback info
def error_message_detail(error, error_detail: sys):

    """
    This function extracts detailed error information
    like file name and line number where error occurred.
    """

    # exc_info() returns 3 values:
    # 1. Exception type
    # 2. Exception value (message)
    # 3. Traceback object (where error occurred)
    # We only need traceback, so we ignore first two using _
    _, _, exc_tb = error_detail.exc_info()

    # Extracting file name from traceback object
    # tb_frame → current execution frame
    # f_code → code object
    # co_filename → gives file name
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Creating a clean and formatted error message
    # {0} → file name
    # {1} → line number
    # {2} → actual error message
    error_message = "Error occurred in python script [{0}] at line number [{1}] with error message [{2}]".format(
        file_name,                # File where error happened
        exc_tb.tb_lineno,         # Line number of error
        str(error)                # Actual error message
    )

    # Returning the final formatted error message
    return error_message



# Creating a custom exception class
# This class inherits from Python's built-in Exception class
class CustomException(Exception):

    """
    Custom Exception class
    Used to generate more detailed error messages
    """

    # Constructor method
    # error_message → actual error message
    # error_detail → system error information
    def __init__(self, error_message, error_detail: sys):

        # Calling parent Exception class constructor
        # This ensures normal exception behavior still works
        super().__init__(error_message)

        # Storing detailed formatted error message
        # We call our custom function to generate full error details
        self.error_message = error_message_detail(
            error_message, 
            error_detail=error_detail
        )

    # This method controls what happens when we print the exception
    def __str__(self):

        # When exception object is printed,
        # this detailed message will be displayed
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e, sys)
    
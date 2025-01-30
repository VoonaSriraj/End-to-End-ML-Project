import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Initialize the base Exception class with the error message
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail):
        # Extract exception info
        exc_type, exc_value, exec_tb = sys.exc_info()

        if exec_tb is None:
            return f"Error: {error_message}"  # Handle cases where traceback is not available

        # Extract the file name and line number from the traceback object
        file_name = exec_tb.tb_frame.f_code.co_filename
        line_number = exec_tb.tb_lineno
        
        return f"Error in file [{file_name}], line [{line_number}]: {error_message}"

    def __str__(self):
        return self.error_message

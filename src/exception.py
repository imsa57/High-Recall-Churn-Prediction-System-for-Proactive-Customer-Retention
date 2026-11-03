import sys, inspect
from src.logger import logger


# def error_message_detail(error,error_detail:sys):
#     _,_,exc_tb = error_detail.exc_info()
#     file_name = exc_tb.tb_frame.f_code.co_filename
#     error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))

#     return error_message



def error_message_detail(error, error_detail: sys):
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        # Get the frame of the caller (where raise was executed)
        caller_frame = inspect.stack()[2]   # [0] = current, [1] = error_message_detail, [2] = caller
        file_name = caller_frame.filename
        line_no = caller_frame.lineno

    error_message = f"Error occurred in python script [{file_name}] line [{line_no}] message [{error}]"
    return error_message



class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details)


    def __str__(self):
        return self.error_message
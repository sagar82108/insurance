import sys
import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occurred in Python script: {file_name}, line number: {exc_tb.tb_lineno}, error message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    logging.basicConfig(
        filename="logs/errors.log",
        format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logging.info("Logging has started")

    

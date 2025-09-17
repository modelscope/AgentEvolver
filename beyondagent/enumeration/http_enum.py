from enum import Enum

class HttpEnum(str, Enum):
    """
    An enumeration for HTTP methods, providing string representations for GET and POST.

    Attributes:
        GET (str): The string representation for the HTTP GET method.
        POST (str): The string representation for the HTTP POST method.
    """
    GET = "get"  # ⭐ Defines the string value for the GET method
    POST = "post"  # ⭐ Defines the string value for the POST method

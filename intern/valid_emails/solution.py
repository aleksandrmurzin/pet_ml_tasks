import re
from typing import List


def valid_emails(strings: List[str]) -> List[str]:
    """Take list of potential emails and returns only valid ones"""

    valid_email_regex = re.compile(r"^[\w+.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", flags=re.A)
    def is_valid_email(email: str):
        return valid_email_regex.fullmatch(email)

    return [i for i in strings if is_valid_email(i)]

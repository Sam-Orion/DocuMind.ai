import re

text = """SuperMart 123
100 Main Street
Anytown, USA 12345
(555) 999-8888

Date: 2023-11-05   Time: 14:30
Trans#: 9876543210
Term: 5

Item                   Price
----------------------------
Organic Bananas         2.99
Almond Milk             4.49
Whole Wheat Bread       3.49
Dark Chocolate          5.99

Subtotal              16.96
Tax (5%)               0.85
Total                 17.81

Payment Method: Visa
Card: ************4321
Auth Code: 1234AB

Rewards ID: 888777666
Points Balance: 540

Return within 30 days with receipt.
Thank you for shopping at SuperMart!"""

# regex: allow multiple separator chars
id_pat = r"(?i)\b(?:Rcpt|Receipt|Trans|Transaction|Trx|Order|Inv|Invoice)\b[ \t]*[:#.]+[ \t]*([A-Z0-9-]{4,})"
match = re.search(id_pat, text)

if match:
    print(f"Matched: '{match.group(0)}'")
    print(f"Group 1: '{match.group(1)}'")
else:
    print("No match found")

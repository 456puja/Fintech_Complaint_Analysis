# This file handles input validation using Pydantic.
# It ensures your API only accepts the correct fields.


# app/schemas.py

from pydantic import BaseModel

class ComplaintRequest(BaseModel):
    complaint_text: str
    product: str
    issue: str

class ComplaintResponse(BaseModel):
    risk_label: int
    risk_probability: float



# Purpose:

#ComplaintRequest → Validates incoming request data

#ComplaintResponse → Standardizes API response format
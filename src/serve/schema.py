from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class SeniorCitizenType(str,Enum):
    Yes = "1"
    No = "0"


class BooleanType(str,Enum):
    Yes = "Yes"
    No = "No"

class ContractType(str,Enum):
   Monthly= "Month-to-month" 
   Yearly = "One year"
   TwoYearly= "Two year"

class PaymentMethodType(str,Enum):
   ElectronicCheck = 'Electronic check'              
   MailedCheck = 'Mailed check'
   BankTransfer = 'Bank transfer (automatic)'   
   CreditCard = 'Credit card (automatic)'

class InternetServiceType(str,Enum):
    DSL = 'DSL' 
    FiberOptic = 'Fiber optic' 
    No = 'No'

class ThreeType(str, Enum):
    No = 'No' 
    Yes='Yes' 
    NoService = 'No internet service'


class PredictionRequest(BaseModel):
    MonthlyCharges: float = Field(..., gt=0, examples=[20.0, 23.0, 45.0], description="Monthly Charges (must be greater than 0)")
    tenure: int = Field(...,examples=[2,4,6,10], description="Till know how many month you have this service" )
    Contract: ContractType
    PaymentMethod: PaymentMethodType
    SeniorCitizen : SeniorCitizenType
    Partner : BooleanType
    Dependents : BooleanType
    InternetService : InternetServiceType
    OnlineSecurity : ThreeType
    OnlineBackup : ThreeType
    DeviceProtection : ThreeType
    StreamingMovies : ThreeType
    PaperlessBilling : BooleanType
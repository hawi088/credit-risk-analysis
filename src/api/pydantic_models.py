from pydantic import BaseModel, Field, validator 
from typing import List 
from datetime import datetime 
 
class TransactionInput(BaseModel): 
    TransactionId: str = Field(...) 
    BatchId: str = Field(...) 
    AccountId: str = Field(...) 
    SubscriptionId: str = Field(...) 
    CustomerId: str = Field(...) 
    CurrencyCode: str = Field("UGX") 
    CountryCode: int = Field(256) 
    ProviderId: str = Field(...) 
    ProductId: str = Field(...) 
    ProductCategory: str = Field(...) 
    ChannelId: str = Field(...) 
    Amount: float = Field(...) 
    Value: int = Field(...) 
    TransactionStartTime: str = Field(...) 
    PricingStrategy: int = Field(...) 
    FraudResult: int = Field(0) 
 
class PredictionRequest(BaseModel): 
    transactions: list 
 
class PredictionResponse(BaseModel): 
    customer_id: str 
    risk_probability: float 
    risk_score: float 
    risk_level: str 
    is_high_risk: bool 
 
class HealthResponse(BaseModel): 
    status: str 
    model_loaded: bool 
    version: str 

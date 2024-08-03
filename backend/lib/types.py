from typing import Literal, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

class TransactionClassificationResult(BaseModel):
    large: bool = False
    rapid: bool = False
    fraud: bool = False

class TransactionParams(BaseModel):
    id: Optional[str] = None 
    time_stamp: int
    from_addr: str
    to_addr: str
    value: float
    method: Optional[int]= None
    class_result: Optional[TransactionClassificationResult] = None
    
class HistTransactions(BaseModel):
    hist: List[TransactionParams]

class Transaction (BaseModel):
    params: TransactionParams
    hist: HistTransactions
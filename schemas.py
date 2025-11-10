"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogpost" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List

# Example schemas (you may keep or remove as needed)

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# AI Politique analysis schema
class Analysis(BaseModel):
    """
    Stores each analysis result for persistence and later retrieval
    Collection name: "analysis"
    """
    input_text: str = Field(..., description="Original text analyzed")
    domain: Optional[str] = Field(None, description="Context domain: politics, legal, humanitarian, etc.")
    language: Optional[str] = Field(None, description="Language of the input text")

    summary: str = Field(..., description="Neutral summary of the text")
    tone: str = Field(..., description="Overall tone classification")
    bias: str = Field(..., description="Detected bias description")
    rhetoric: str = Field(..., description="Dominant rhetorical style")
    strategy: str = Field(..., description="Communication/argument strategy")

    keywords: List[str] = Field(default_factory=list, description="Key terms extracted from the text")
    recommendations: List[str] = Field(default_factory=list, description="3 recommendations/interpretations")

    source: Optional[str] = Field(None, description="Where the text came from (manual, YouTube, X, etc.)")
    user_id: Optional[str] = Field(None, description="Optional user identifier if auth is added")

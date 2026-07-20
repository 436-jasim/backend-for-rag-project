from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr

# Import your helper functions from the shared security package
from shared.security import (
    hash_password,
    verify_password,
    create_access_token,
    verify_token
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Setup OAuth2 scheme to extract 'Bearer <token>' headers automatically
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# In-memory mock database (Replace with SQLite/PostgreSQL models in production)
# Structure: { "username": {"id": "123", "username": "...", "email": "...", "password": "hashed..."} }
# PROBLEM: Using an in-memory `db_users` while signup writes to MongoDB `users_collection`.
# db_users: Dict[str, dict] = {}

# ---------------------------------------------------
# Request Schemas
# ---------------------------------------------------
class UserSignup(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: str

# ---------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------
# At the top of auth_routes.py:
from shared.database import users_collection, chats_collection, messages_collection

# In your signup route:
@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup):
    existing_user = await users_collection.find_one({"username": user_data.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists.")
    # Hash the password
    hashed_pwd = hash_password(user_data.password)

    # Prepare user document WITHOUT a user_id; we'll set a canonical user_id after insert
    new_user = {
        "username": user_data.username,
        "email": user_data.email,
        "password": hashed_pwd
    }

    # Insert and then assign a canonical user_id equal to the inserted ObjectId string
    result = await users_collection.insert_one(new_user)
    try:
        canonical_id = str(result.inserted_id)
        # Persist canonical user_id on the document for later lookups and session ids
        await users_collection.update_one({"_id": result.inserted_id}, {"$set": {"user_id": canonical_id}})
    except Exception as e:
        # If updating the document fails, keep going but log the warning (email/username remain)
        print("Warning: failed to set canonical user_id:", e)

    return {"status": "success", "message": "User registered successfully!"}


@router.post("/login")
async def login(credentials: UserLogin):
    """Authenticates credentials and returns a signed JWT access token."""
    user = await users_collection.find_one({"$or": [{"username": credentials.username}, {"email": credentials.username}]})

    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    canonical_user_id = str(user.get("_id"))
    legacy_user_id = user.get("user_id")

    if legacy_user_id and legacy_user_id != canonical_user_id:
        await users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"user_id": canonical_user_id}}
        )
        await chats_collection.update_many(
            {"user_id": legacy_user_id},
            {"$set": {"user_id": canonical_user_id}}
        )
        await messages_collection.update_many(
            {"user_id": legacy_user_id},
            {"$set": {"user_id": canonical_user_id}}
        )
    elif not legacy_user_id:
        await users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"user_id": canonical_user_id}}
        )

    access_token = create_access_token(data={"sub": canonical_user_id})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": canonical_user_id,
        "session_id": canonical_user_id,
    }


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Dependency: Decodes JWT token and retrieves current user profile."""
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # PROBLEM: original code read from the in-memory `db_users` which is not populated.
    # FIX: query MongoDB for the user record by canonical `user_id`.
    user = await users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User profile not found.")

    return user


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Protected Endpoint: Returns current authenticated user details."""
    return {
        "username": current_user["username"],
        "email": current_user["email"]
    }
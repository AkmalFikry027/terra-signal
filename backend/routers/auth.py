from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

import config
from services import security

router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
    row = security.user_from_token(creds.credentials) if creds else None
    if row is None:
        raise HTTPException(401, "Not authenticated or session expired.",
                            headers={"WWW-Authenticate": "Bearer"})
    return row


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordIn(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


def _session(row):
    token, ttl = security.create_token(row)
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl,
            "user": security.public_user(row)}


@router.post("/login")
def login(body: LoginIn, request: Request):
    ip = request.client.host if request.client else "unknown"
    wait = security.lock_remaining(body.username, ip)
    if wait:
        raise HTTPException(429, f"Too many failed attempts. Try again in {wait} seconds.")
    row = security.authenticate(body.username.strip(), body.password, ip)
    if row is None:
        raise HTTPException(401, "Incorrect username or password.")
    return _session(row)


@router.get("/me")
def me(user=Depends(current_user)):
    return security.public_user(user)


@router.post("/change-password")
def change_password(body: ChangePasswordIn, user=Depends(current_user)):
    if not security.verify_password(body.current_password, user["password_hash"]):
        raise HTTPException(400, "Current password is incorrect.")
    if len(body.new_password) < config.MIN_PASSWORD_LENGTH:
        raise HTTPException(422, f"New password must be at least {config.MIN_PASSWORD_LENGTH} characters.")
    if body.new_password == body.current_password:
        raise HTTPException(422, "New password must be different from the current password.")
    if body.new_password.lower() == user["username"].lower():
        raise HTTPException(422, "New password must not be the same as your username.")
    updated = security.change_password(user, body.new_password)
    return {"message": "Password updated.", **_session(updated)}  # fresh token keeps this session alive

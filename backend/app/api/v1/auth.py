"""认证 API — 注册 / 登录."""

from fastapi import APIRouter, HTTPException
from ...schemas.auth import UserRegister, UserLogin, TokenResponse, UserInfo
from ...models.user import user_db
from ...core.security import verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register", response_model=TokenResponse)
async def register(body: UserRegister):
    ok = user_db.create_user(body.username, body.password, body.email)
    if not ok:
        raise HTTPException(400, "用户名已存在")
    token = create_access_token({"sub": body.username, "role": "user"})
    return TokenResponse(access_token=token, username=body.username, role="user")


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    user = user_db.get_user(body.username)
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(401, "用户名或密码错误")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return TokenResponse(access_token=token, username=user["username"], role=user["role"])

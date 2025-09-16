"""
Authentication management for Brokle SDK.
"""

import httpx
from typing import Optional, Dict, Any
from pydantic import BaseModel

from .config import Config


class AuthInfo(BaseModel):
    """Authentication information."""
    
    public_key: str
    secret_key: str
    environment: str
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    tier: Optional[str] = None
    permissions: Optional[Dict[str, Any]] = None


class AuthManager:
    """Manages authentication with Brokle Platform."""
    
    def __init__(self, config: Config):
        self.config = config
        self._auth_info: Optional[AuthInfo] = None
        self._validated = False
    
    async def validate_public_key(self) -> AuthInfo:
        """Validate public key with the platform."""
        if not self.config.public_key:
            raise ValueError("Public key is required")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.config.host}/api/v1/auth/validate",
                    headers=self.config.get_headers(),
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                if not data.get('success'):
                    raise ValueError(f"Public key validation failed: {data.get('error', 'Unknown error')}")
                
                auth_data = data.get('data', {})
                self._auth_info = AuthInfo(
                    public_key=self.config.public_key,
                    secret_key=auth_data.get('secret_key', self.config.secret_key),
                    environment=auth_data.get('environment', self.config.environment),
                    organization_id=auth_data.get('organization_id'),
                    user_id=auth_data.get('user_id'),
                    tier=auth_data.get('tier'),
                    permissions=auth_data.get('permissions')
                )
                
                self._validated = True
                return self._auth_info
                
            except httpx.HTTPError as e:
                raise ValueError(f"Failed to validate public key: {e}")
    
    def get_auth_info(self) -> Optional[AuthInfo]:
        """Get cached authentication information."""
        return self._auth_info
    
    def is_validated(self) -> bool:
        """Check if public key has been validated."""
        return self._validated
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return self.config.get_headers()
    
    def clear_auth(self) -> None:
        """Clear authentication information."""
        self._auth_info = None
        self._validated = False
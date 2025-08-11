from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional

import requests


class JiraClient:
    """Minimal Jira Cloud REST v3 client.

    Configure via env:
    - JIRA_BASE_URL: e.g., https://your-domain.atlassian.net
    - JIRA_EMAIL: Atlassian account email
    - JIRA_API_TOKEN: API token from https://id.atlassian.com/manage-profile/security/api-tokens
    - JIRA_PROJECT_KEY: e.g., STOCK
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        project_key: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("JIRA_BASE_URL", "")).rstrip("/")
        self.email = email or os.getenv("JIRA_EMAIL", "")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN", "")
        self.project_key = project_key or os.getenv("JIRA_PROJECT_KEY", "")

        if not all([self.base_url, self.email, self.api_token, self.project_key]):
            raise ValueError("Missing Jira configuration env vars: JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY")

        token = f"{self.email}:{self.api_token}".encode("utf-8")
        self._auth_header = {
            "Authorization": f"Basic {base64.b64encode(token).decode('utf-8')}",
            "Accept": "application/json",
        }

    def _get_json(self, path: str, timeout: int = 10):
        url = f"{self.base_url}{path}"
        res = requests.get(url, headers=self._auth_header, timeout=timeout)
        # Raise for HTTP error first for clarity
        try:
            res.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"HTTP {res.status_code} {res.reason} for {path}: {res.text[:200]}") from e
        # Ensure JSON response
        ct = (res.headers.get("Content-Type") or "").lower()
        if "json" not in ct:
            raise RuntimeError(f"Non-JSON response for {path}: {res.text[:200]}")
        try:
            return res.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON for {path}: {res.text[:200]}") from e

    def test_connection(self) -> Dict[str, Any]:
        """Validate credentials and project access without creating an issue."""
        try:
            me = self._get_json("/rest/api/3/myself")
            proj = self._get_json(f"/rest/api/3/project/{self.project_key}")
            return {"status": "success", "accountId": me.get("accountId"), "project": proj.get("key")}
        except Exception as e:
            hint = "Check JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, and JIRA_PROJECT_KEY are correct and for Jira Cloud."
            return {"status": "error", "error": f"{e}", "hint": hint}

    def create_issue(
        self,
        summary: str,
        description: str,
        issue_type: str = "Task",
        labels: Optional[list[str]] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/rest/api/3/issue"
        payload: Dict[str, Any] = {
            "fields": {
                "project": {"key": self.project_key},
                "summary": summary[:255],
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {"type": "paragraph", "content": [{"type": "text", "text": description[:30000]}]}
                    ],
                },
                "issuetype": {"name": issue_type},
            }
        }
        if labels:
            payload["fields"]["labels"] = labels
        if extra_fields:
            payload["fields"].update(extra_fields)

        res = requests.post(url, headers={**self._auth_header, "Content-Type": "application/json"}, data=json.dumps(payload), timeout=15)
        res.raise_for_status()
        return res.json()

    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}/comment"
        payload = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment[:30000]}]}],
            }
        }
        res = requests.post(url, headers={**self._auth_header, "Content-Type": "application/json"}, data=json.dumps(payload), timeout=15)
        res.raise_for_status()
        return res.json()

    def attach_file(self, issue_key: str, file_path: str) -> Dict[str, Any]:
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}/attachments"
        headers = {**self._auth_header, "X-Atlassian-Token": "no-check"}
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
            res = requests.post(url, headers=headers, files=files, timeout=30)
        res.raise_for_status()
        return res.json()


def safe_create_issue(summary: str, description: str, **kwargs) -> Dict[str, Any]:
    """Best-effort issue creation. Returns dict with status and optional data/error."""
    try:
        client = JiraClient(
            base_url=kwargs.get("base_url"),
            email=kwargs.get("email"),
            api_token=kwargs.get("api_token"),
            project_key=kwargs.get("project_key"),
        )
        data = client.create_issue(summary, description, issue_type=kwargs.get("issue_type", "Task"), labels=kwargs.get("labels"))
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def safe_test_connection(**kwargs) -> Dict[str, Any]:
    """Best-effort connectivity test to Jira."""
    try:
        client = JiraClient(
            base_url=kwargs.get("base_url"),
            email=kwargs.get("email"),
            api_token=kwargs.get("api_token"),
            project_key=kwargs.get("project_key"),
        )
        return client.test_connection()
    except Exception as e:
        return {"status": "error", "error": str(e)}



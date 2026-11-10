from __future__ import annotations

import argparse
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Authorize the Google account used by FrameBucket's Drive storage."
    )
    parser.add_argument(
        "--client-secrets",
        required=True,
        type=Path,
        help="Path to the OAuth Desktop client JSON downloaded from Google Cloud.",
    )
    args = parser.parse_args()
    if not args.client_secrets.is_file():
        parser.error(f"OAuth client file not found: {args.client_secrets}")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(args.client_secrets), scopes=[DRIVE_SCOPE]
    )
    credentials = flow.run_local_server(
        host="localhost",
        port=0,
        access_type="offline",
        prompt="consent",
    )
    if not credentials.refresh_token:
        raise RuntimeError(
            "Google did not return a refresh token. Revoke this app's access in your Google account and run the command again."
        )

    print("Google Drive OAuth completed. Keep the refresh token private.")
    print(f"FB_GOOGLE_CLIENT_ID={credentials.client_id}")
    print(f"FB_GOOGLE_CLIENT_SECRET={credentials.client_secret}")
    print(f"FB_GOOGLE_REFRESH_TOKEN={credentials.refresh_token}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Publish daily picks to various public hosting services
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

def publish_to_github_gist(picks_data: Dict, token: str) -> Optional[str]:
    """Publish picks to GitHub Gist (public)"""
    try:
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        payload = {
            'description': f'Stock4U Daily Picks - {datetime.now().strftime("%Y-%m-%d")}',
            'public': True,
            'files': {
                'daily_picks.json': {
                    'content': json.dumps(picks_data, indent=2)
                }
            }
        }
        
        response = requests.post('https://api.github.com/gists', 
                               headers=headers, 
                               json=payload)
        
        if response.ok:
            gist_data = response.json()
            raw_url = gist_data['files']['daily_picks.json']['raw_url']
            print(f"✅ Published to GitHub Gist: {raw_url}")
            return raw_url
        else:
            print(f"❌ GitHub Gist failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ GitHub Gist error: {e}")
        return None

def publish_to_pastebin(picks_data: Dict, api_key: str) -> Optional[str]:
    """Publish picks to Pastebin"""
    try:
        payload = {
            'api_dev_key': api_key,
            'api_option': 'paste',
            'api_paste_code': json.dumps(picks_data, indent=2),
            'api_paste_name': f'Stock4U Daily Picks - {datetime.now().strftime("%Y-%m-%d")}',
            'api_paste_format': 'json',
            'api_paste_private': '0',  # Public
            'api_paste_expire_date': '1W'  # 1 week
        }
        
        response = requests.post('https://pastebin.com/api/api_post.php', data=payload)
        
        if response.ok and response.text.startswith('https://'):
            paste_url = response.text.strip()
            # Convert to raw URL
            raw_url = paste_url.replace('pastebin.com/', 'pastebin.com/raw/')
            print(f"✅ Published to Pastebin: {raw_url}")
            return raw_url
        else:
            print(f"❌ Pastebin failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Pastebin error: {e}")
        return None

def publish_to_jsonbin(picks_data: Dict, api_key: str) -> Optional[str]:
    """Publish picks to JSONBin.io"""
    try:
        headers = {
            'X-Master-Key': api_key,
            'Content-Type': 'application/json'
        }
        
        response = requests.post('https://api.jsonbin.io/v3/b', 
                               headers=headers, 
                               json=picks_data)
        
        if response.ok:
            bin_data = response.json()
            bin_id = bin_data['metadata']['id']
            raw_url = f"https://api.jsonbin.io/v3/b/{bin_id}/latest"
            print(f"✅ Published to JSONBin: {raw_url}")
            return raw_url
        else:
            print(f"❌ JSONBin failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ JSONBin error: {e}")
        return None

def publish_daily_picks(picks_file: str = "cache/daily_picks.json") -> Optional[str]:
    """Publish daily picks to a public URL"""
    
    # Load picks data
    picks_path = Path(picks_file)
    if not picks_path.exists():
        print(f"❌ Picks file not found: {picks_file}")
        return None
    
    try:
        with open(picks_path, 'r') as f:
            picks_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load picks: {e}")
        return None
    
    # Try different services in order of preference
    
    # 1. GitHub Gist (if token available)
    github_token = os.getenv('GITHUB_TOKEN')
    if github_token:
        url = publish_to_github_gist(picks_data, github_token)
        if url:
            return url
    
    # 2. JSONBin.io (if API key available)
    jsonbin_key = os.getenv('JSONBIN_API_KEY')
    if jsonbin_key:
        url = publish_to_jsonbin(picks_data, jsonbin_key)
        if url:
            return url
    
    # 3. Pastebin (if API key available)
    pastebin_key = os.getenv('PASTEBIN_API_KEY')
    if pastebin_key:
        url = publish_to_pastebin(picks_data, pastebin_key)
        if url:
            return url
    
    print("❌ No API keys found. Set one of:")
    print("  - GITHUB_TOKEN (recommended)")
    print("  - JSONBIN_API_KEY")
    print("  - PASTEBIN_API_KEY")
    
    return None

if __name__ == "__main__":
    # Generate picks first
    from utils.daily_picks import run_daily_picks_job
    
    print("🔄 Generating daily picks...")
    run_daily_picks_job()
    
    print("📤 Publishing to public URL...")
    url = publish_daily_picks()
    
    if url:
        print(f"\n🎉 SUCCESS! Your daily picks are now available at:")
        print(f"   {url}")
        print(f"\n📋 Set this in your Streamlit Cloud secrets:")
        print(f'   DAILY_PICKS_URL = "{url}"')
    else:
        print("\n❌ Failed to publish. Check your API keys.")

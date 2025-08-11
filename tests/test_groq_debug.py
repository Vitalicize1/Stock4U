#!/usr/bin/env python3
"""
Debug script to test Groq API directly and see what the actual error is.
"""

import os
from dotenv import load_dotenv
import requests
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def test_groq_api_directly():
    """Test Groq API directly to see what the actual error is."""
    
    print("🔍 Testing Groq API directly...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment variables")
        return
    
    print(f"✅ Found API key: {api_key[:10]}...")
    
    # Test the API directly
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "user",
                "content": "Hello, this is a test message."
            }
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    try:
        print("📡 Making API call...")
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ API call successful!")
            response_data = response.json()
            print(f"📝 Response: {response_data}")
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"📝 Error Response: {response.text}")
            
            # Try to parse error details
            try:
                error_data = response.json()
                print(f"🔍 Error Details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"🔍 Raw Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")

def test_langchain_groq():
    """Test LangChain Groq client."""
    
    print("\n🔍 Testing LangChain Groq client...")
    
    try:
        from langchain_groq import ChatGroq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY not found")
            return
        
        print("📡 Creating LangChain client...")
        client = ChatGroq(
            model_name="llama3-8b-8192",
            groq_api_key=api_key,
            temperature=0.1,
            max_tokens=100
        )
        
        print("📡 Making test call...")
        response = client.invoke("Hello, this is a test.")
        print(f"✅ LangChain call successful: {response}")
        
    except Exception as e:
        print(f"❌ LangChain test failed: {str(e)}")
        print(f"🔍 Error type: {type(e)}")

if __name__ == "__main__":
    test_groq_api_directly()
    test_langchain_groq()

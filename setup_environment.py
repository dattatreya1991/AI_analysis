#!/usr/bin/env python3
"""
Setup script for configuring the GEMINI_API_KEY environment variable
"""

import os
import sys

def setup_gemini_api_key():
    """
    Interactive setup for GEMINI_API_KEY environment variable
    """
    print("Google Gemini API Key Setup")
    print("=" * 40)
    
    # Check if already set
    current_key = os.getenv("GEMINI_API_KEY")
    if current_key:
        print(f"GEMINI_API_KEY is already set: {current_key[:10]}...{current_key[-4:]}")
        update = input("Do you want to update it? (y/N): ").lower().strip()
        if update != 'y':
            print("Setup cancelled.")
            return
    
    print("\nTo get your Gemini API key:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated key")
    
    api_key = input("\n Enter your Gemini API key: ").strip()
    
    if not api_key:
        print(" No API key provided. Setup cancelled.")
        return
    
    # Set environment variable for current session
    os.environ["GEMINI_API_KEY"] = api_key
    
    # Instructions for permanent setup
    print("\nAPI key set for current session!")
    print("\nTo make this permanent, add this line to your shell profile:")
    print(f"export GEMINI_API_KEY='{api_key}'")
    print("\nFor bash: ~/.bashrc or ~/.bash_profile")
    print("For zsh: ~/.zshrc")
    print("For fish: ~/.config/fish/config.fish")
    
    # Test the API key
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello, this is a test.")
        print("\nAPI key is working correctly!")
        print(f"Test response: {response.text[:100]}...")
    except Exception as e:
        print(f"\nWarning: Could not verify API key: {str(e)}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    setup_gemini_api_key()


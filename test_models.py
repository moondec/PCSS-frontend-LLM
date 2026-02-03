#!/usr/bin/env python3
"""
Quick script to list all available models from PCSS API
"""
import sys
sys.path.insert(0, '/Users/marekurbaniak/Documents/Bielik')

from pcss_llm_app.config import ConfigManager
from pcss_llm_app.core.api_client import PcssApiClient

def main():
    config = ConfigManager()
    api = PcssApiClient(config)
    
    if not api.is_configured():
        print("ERROR: API not configured. Please set API key first.")
        return
    
    try:
        models = api.list_models()
        print(f"\n✅ Found {len(models)} models:")
        print("=" * 60)
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

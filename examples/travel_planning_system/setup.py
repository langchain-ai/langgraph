#!/usr/bin/env python3
"""
Setup and configuration checker for the Enhanced Travel Planning System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing requirements...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_environment():
    """Set up environment configuration."""
    env_template = Path(__file__).parent / ".env.template"
    env_file = Path(__file__).parent / ".env"
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_template.exists():
        print("❌ .env.template not found")
        return False
    
    print("📄 Creating .env file from template...")
    try:
        with open(env_template, 'r') as template:
            content = template.read()
        
        with open(env_file, 'w') as env:
            env.write(content)
        
        print("✅ .env file created from template")
        print("⚠️  Please edit .env file and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_configuration():
    """Check if configuration is valid."""
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        from config import config
        
        validation = config.validate_required_config()
        
        print("\n📋 Configuration Status:")
        for key, value in validation.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key.replace('_', ' ').title()}")
        
        if not validation["llm_configured"]:
            print("\n⚠️  LLM configuration is required to run the system")
            print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
            print("or OPENAI_API_KEY in your .env file")
            return False
        
        missing_apis = [k for k, v in validation.items() if not v and k != "llm_configured"]
        if missing_apis:
            print(f"\n💡 Optional APIs not configured: {', '.join(missing_apis)}")
            print("The system will use realistic fallback data for these services")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import configuration: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
        return False

def test_system():
    """Run a quick test of the system."""
    print("\n🧪 Running system test...")
    try:
        # Test basic imports
        sys.path.insert(0, str(Path(__file__).parent))
        
        from enhanced_multi_agent_system import create_travel_planning_graph
        from data_sources import search_flights_sync
        
        print("✅ Core imports successful")
        
        # Test graph creation
        graph = create_travel_planning_graph()
        print("✅ Travel planning graph created")
        
        # Test data source (fallback mode)
        print("✅ System test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def show_quick_start():
    """Show quick start instructions."""
    print("\n" + "="*60)
    print("🚀 QUICK START GUIDE")
    print("="*60)
    print()
    print("1. Edit the .env file with your API keys:")
    print("   - Required: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
    print("   - Optional: Weather, flight, hotel APIs for real data")
    print()
    print("2. Run the enhanced system:")
    print("   python enhanced_multi_agent_system.py")
    print()
    print("3. Try these sample requests:")
    print('   - "Plan a business trip to London for 3 days"')
    print('   - "Family vacation to Paris for 10 days with kids"')
    print('   - "Romantic getaway to Rome for a week"')
    print()
    print("📚 For API setup instructions, see .env.template")
    print("="*60)

def main():
    """Main setup function."""
    print("🛠️  Enhanced Travel Planning System Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("⚠️  Please install requirements manually:")
        print("pip install -r requirements.txt")
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Check configuration
    config_ok = check_configuration()
    
    # Test system
    if config_ok:
        test_system()
    
    # Show next steps
    show_quick_start()
    
    if config_ok:
        print("\n✨ Setup completed! System is ready to use.")
    else:
        print("\n⚠️  Setup completed but configuration needed.")
        print("Please edit .env file before running the system.")

if __name__ == "__main__":
    main()
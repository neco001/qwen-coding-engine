#!/usr/bin/env python3
"""
Activation script for Production Blocking Mode.

This script enables production blocking in the Anti-Degradation System
by updating the configuration, creating a git tag, and verifying readiness.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

CONFIG_PATH = Path(".anti_degradation/config.yaml")
GIT_TAG_PREFIX = "anti-degradation/prod-block-"

class ActivationError(Exception):
    """Custom exception for activation failures."""
    pass

def load_config() -> Dict[str, Any]:
    """Load the current configuration file."""
    if not CONFIG_PATH.exists():
        raise ActivationError(f"Configuration file not found: {CONFIG_PATH}")
    
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any]) -> None:
    """Save the updated configuration file."""
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def update_production_mode(config: Dict[str, Any]) -> Dict[str, Any]:
    """Update the production_mode.enabled setting to true."""
    if 'production_mode' not in config:
        config['production_mode'] = {}
    
    if 'enabled' in config['production_mode'] and config['production_mode']['enabled']:
        print("⚠️  Production blocking is already enabled.")
        return config
    
    config['production_mode']['enabled'] = True
    config['production_mode']['activated_at'] = datetime.utcnow().isoformat()
    print("✅ Updated production_mode.enabled to true")
    return config

def run_git_command(args: list) -> str:
    """Run a git command and return stdout."""
    try:
        result = subprocess.run(
            ['git'] + args,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise ActivationError(f"Git command failed: {' '.join(args)}\n{e.stderr}")

def create_activation_tag() -> str:
    """Create a git tag for the activation milestone."""
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    tag_name = f"{GIT_TAG_PREFIX}{timestamp}"
    
    # Check for uncommitted changes
    status = run_git_command(['status', '--porcelain'])
    if status:
        raise ActivationError(
            "Cannot create activation tag with uncommitted changes. "
            "Please commit config changes first."
        )
    
    run_git_command(['tag', '-a', tag_name, '-m', f'Activate Production Blocking {timestamp}'])
    print(f"✅ Created git tag: {tag_name}")
    return tag_name

def verify_workflow_exists() -> bool:
    """Verify the production workflow file exists."""
    workflow_path = Path(".github/workflows/anti_degradation_production.yml")
    if not workflow_path.exists():
        print("⚠️  Warning: Production workflow file not found.")
        print(f"   Expected: {workflow_path}")
        return False
    print("✅ Production workflow file verified")
    return True

def print_activation_summary(tag_name: str, config: Dict[str, Any]) -> None:
    """Print a summary of the activation."""
    print("\n" + "="*60)
    print("ANTI-DEGRADATION PRODUCTION BLOCKING ACTIVATED")
    print("="*60)
    print(f"Activation Time : {config['production_mode'].get('activated_at')}")
    print(f"Git Tag         : {tag_name}")
    print(f"Config Path     : {CONFIG_PATH}")
    print("-"*60)
    print("NEXT STEPS:")
    print("1. Commit and push the config changes")
    print("2. Push the tag: git push origin {tag_name}")
    print("3. Monitor the next main branch push for blocking behavior")
    print("="*60 + "\n")

def main():
    """Main entry point for activation script."""
    print("🚀 Starting Production Blocking Activation...")
    
    try:
        # 1. Load and update config
        config = load_config()
        config = update_production_mode(config)
        save_config(config)
        
        # 2. Verify workflow existence
        workflow_exists = verify_workflow_exists()
        
        # 3. Create git tag
        tag_name = create_activation_tag()
        
        # 4. Print summary
        print_activation_summary(tag_name, config)
        
        sys.exit(0)
        
    except ActivationError as e:
        print(f"❌ Activation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()